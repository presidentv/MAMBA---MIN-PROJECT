"""Tests for the changes that make the pipeline run on the full DAiSEE release.

Each one guards a failure that the 216-clip development subset never exercised
but a ~9,000-clip unattended run would hit: an overflow at large n, one corrupt
video stopping everything, a half-written cache file counted as done, a
significance test that is only valid on balanced data.

Run:  python -m pytest tests -q
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_finetune import binomial_p  # noqa: E402
from src.dataset import _enforce_missing_policy  # noqa: E402
from src.preprocess import _atomic_savez, _remove_stale_partials  # noqa: E402
from src.utils import load_config, missing_clip_policy  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------- significance
def test_binomial_matches_the_exact_sum_at_small_n():
    from math import comb
    n, k, p0 = 80, 35, 0.25
    exact = sum(comb(n, i) * p0**i * (1 - p0)**(n - i) for i in range(k, n + 1))
    assert binomial_p(k, n, p0) == pytest.approx(exact, rel=1e-9)


def test_binomial_survives_full_test_split_size():
    # The exact comb() sum raises OverflowError here; the full DAiSEE test split
    # is ~1,784 clips, so this is the size the final report is computed at.
    p = binomial_p(900, 1784, 0.25)
    assert 0.0 <= p < 1e-50


def test_binomial_edge_cases():
    assert binomial_p(0, 10, 0.3) == pytest.approx(1.0)
    assert binomial_p(0, 0, 0.3) == 1.0


# -------------------------------------------------------------- missing clips
def _cfg(policy=None, frac=None):
    ds = {}
    if policy is not None:
        ds["missing_clip_policy"] = policy
    if frac is not None:
        ds["max_missing_fraction"] = frac
    return {"dataset": ds}


def test_default_policy_is_strict():
    assert missing_clip_policy(_cfg()) == (False, 0.0)
    with pytest.raises(FileNotFoundError):
        _enforce_missing_policy(_cfg(), "train", 100, ["a.avi"], None, "cache")


def test_skip_policy_tolerates_a_few_bad_clips():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _enforce_missing_policy(_cfg("skip", 0.02), "train", 1000, ["a.avi"] * 5, None, "cache")
    assert any("skipping 5/1000" in str(w.message) for w in caught)


def test_skip_policy_still_stops_an_incomplete_preprocessing_run():
    # 30% missing is not "a few corrupt videos"; training on the rest would be
    # silently wrong.
    with pytest.raises(FileNotFoundError, match="above dataset.max_missing_fraction"):
        _enforce_missing_policy(_cfg("skip", 0.02), "train", 100, ["x"] * 30, None, "cache")


def test_explicit_require_all_overrides_the_config():
    with pytest.raises(FileNotFoundError):
        _enforce_missing_policy(_cfg("skip", 0.5), "val", 100, ["x"], True, "cache")
    _enforce_missing_policy(_cfg(), "val", 100, ["x"] * 90, False, "cache")


def test_bad_policy_values_are_rejected():
    with pytest.raises(ValueError):
        missing_clip_policy(_cfg("ignore"))
    with pytest.raises(ValueError):
        missing_clip_policy(_cfg("skip", 1.5))


# ------------------------------------------------------------------- caching
def test_atomic_savez_leaves_no_partial_file(tmp_path):
    out = tmp_path / "clip.npz"
    _atomic_savez(out, {"a": np.arange(5), "b": np.ones((2, 3))})
    assert out.exists()
    assert not list(tmp_path.glob("*.partial"))
    with np.load(out) as d:
        assert d["a"].tolist() == [0, 1, 2, 3, 4]


def test_atomic_savez_keeps_the_npz_name(tmp_path):
    # np.savez_compressed appends ".npz" to a path without it; writing through a
    # file handle must not produce "clip.npz.partial.npz".
    out = tmp_path / "clip.npz"
    _atomic_savez(out, {"a": np.zeros(1)})
    assert sorted(p.name for p in tmp_path.iterdir()) == ["clip.npz"]


def test_stale_partial_cleanup_only_touches_its_own_clip(tmp_path):
    # A split-wide sweep deleted other shards' in-progress writes (and raised
    # PermissionError on Windows). Cleanup must be scoped to one clip.
    mine = tmp_path / "111.npz"
    (tmp_path / "111.npz.4242.partial").write_bytes(b"x")
    other = tmp_path / "222.npz.5151.partial"
    other.write_bytes(b"y")
    _remove_stale_partials(mine)
    assert not (tmp_path / "111.npz.4242.partial").exists()
    assert other.exists()


def test_shard_partition_covers_every_clip_exactly_once():
    records = list(range(103))
    n = 8
    shards = [records[i::n] for i in range(n)]
    flat = sorted(x for s in shards for x in s)
    assert flat == records


# -------------------------------------------------------------------- config
def test_env_overrides_the_dataset_root(monkeypatch, tmp_path):
    monkeypatch.setenv("DAISEE_ROOT", str(tmp_path))
    cfg = load_config(REPO / "configs" / "config_full.yaml")
    assert cfg["paths"]["dataset_root"] == str(tmp_path)
    assert "DAISEE_ROOT" in cfg["_path_overrides"]["dataset_root"]


def test_full_config_is_consistent():
    cfg = load_config(REPO / "configs" / "config_full.yaml")
    assert cfg["video"]["num_frames"] == 32
    assert cfg["vit"]["freeze"] is False
    # Off for the 16 GB RTX A4000 the full run targets; either value is valid,
    # but it must be an explicit boolean, not left to the code default.
    assert isinstance(cfg["vit"]["grad_checkpointing"], bool)
    assert missing_clip_policy(cfg) == (True, 0.02)
    # Must not share a calibration file with the development runs.
    dev = load_config(REPO / "configs" / "config_ft32.yaml")
    assert cfg["mrs"]["calibration_file"] != dev["mrs"]["calibration_file"]
    assert cfg["training"]["class_weighting"] == "effective_number"
    # Effective batch 32 by accumulation; 4 clips x 32 frames per forward is the
    # size measured to fit a 6-8 GB card with gradient checkpointing, and
    # estimated at ~9-10 GB without it.
    tr = cfg["training"]
    assert tr["batch_size"] * tr["grad_accum_steps"] == 32
    assert tr["batch_size"] * cfg["video"]["num_frames"] <= 128
    # Learning rates are the development values x sqrt(32 / 8) = 2.
    assert tr["learning_rate"] == 2 * dev["training"]["learning_rate"]
    assert cfg["vit"]["finetune_lr"] == 2 * dev["vit"]["finetune_lr"]
    assert tr["warmup_epochs"] < tr["extend_decision_epoch"] < tr["epochs"]


# ------------------------------------------------------------ decision point
def test_extend_only_when_still_rising():
    from src.finetune import should_extend
    rising = [0.20 + 0.01 * i for i in range(30)]
    peaked = [0.30] * 10 + [0.40] + [0.35] * 19          # best at epoch 11
    flat_tail = [0.20 + 0.01 * i for i in range(25)] + [0.443] * 5
    assert should_extend(rising, window=5, min_delta=0.005)
    assert not should_extend(peaked, window=5, min_delta=0.005)
    # Last-window best 0.443 vs 0.44 before: within the margin, so not rising.
    assert not should_extend(flat_tail, window=5, min_delta=0.005)
    assert not should_extend([0.3] * 5, window=5, min_delta=0.005)


class _Terminal:
    """A stdin stand-in: replays typed lines, or blocks like an unattended one."""

    def __init__(self, lines=None, tty=True):
        self.lines, self.tty = list(lines or []), tty

    def isatty(self):
        return self.tty

    def readline(self):
        if self.lines is None:
            import threading
            threading.Event().wait(5)          # nobody at the keyboard
            return ""
        return self.lines.pop(0) if self.lines else ""


@pytest.mark.parametrize("typed, expected", [
    (["y\n"], (True, "yes")), (["NO\n"], (False, "no")),
    (["maybe\n", "\n", "yes\n"], (True, "yes")),        # re-asks until y or n
    ([], (True, "no answer (timeout)")),                # end of input -> default
])
def test_ask_to_continue_answers(typed, expected, capsys):
    from src.finetune import ask_to_continue
    assert ask_to_continue("still rising", 5, True, stream=_Terminal(typed)) == expected
    assert "Continue? [y/n]" in capsys.readouterr().out


def test_ask_to_continue_times_out_and_skips_without_terminal():
    import time
    from src.finetune import ask_to_continue
    t0 = time.monotonic()
    t = _Terminal()
    t.lines = None
    assert ask_to_continue("q", 0.3, True, stream=t) == (True, "no answer (timeout)")
    assert time.monotonic() - t0 < 3
    assert ask_to_continue("q", 60, False, stream=_Terminal(tty=False)) == \
        (False, "not asked (no terminal)")


def test_periodic_snapshot_names():
    from src.finetune import periodic_checkpoint_name
    saved = [periodic_checkpoint_name(e, 10) for e in range(64)]
    assert [s for s in saved if s] == [f"after_epoch_{n:03d}.pt" for n in (10, 20, 30, 40, 50, 60)]
    assert periodic_checkpoint_name(9, 0) is None


def test_two_cycle_schedule():
    from src.finetune import make_lr_lambda
    lam = make_lr_lambda(epochs=64, warmup=3, scheduler="cosine",
                         decision_epoch=30, restart_factor=0.5)
    assert lam(0) == pytest.approx(1 / 3)
    assert lam(3) == pytest.approx(1.0)
    # The first cycle is nearly annealed by the decision point ...
    assert lam(29) < 0.01
    # ... and a continued run restarts at half the peak, then anneals again.
    assert lam(30) == pytest.approx(0.5)
    assert lam(63) < 0.005
    # Without a decision epoch it is the original single cosine.
    single = make_lr_lambda(epochs=64, warmup=3, scheduler="cosine",
                            decision_epoch=None, restart_factor=0.5)
    assert single(30) == pytest.approx(0.5 * (1 + np.cos(np.pi * 27 / 61)))
