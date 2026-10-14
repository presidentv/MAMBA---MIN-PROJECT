"""Tests for collapsing the engagement levels into fewer classes.

The failure this guards against is not a crash. It is a run where the training
split is merged and the test split is not, or where a 3-class model's report is
labelled with the four original level names. Both produce plausible-looking
numbers that mean nothing, so the tests pin the properties rather than the
mechanics: the partition is total, the remap is applied to every split
identically, the original label survives, and the names follow the labelling.

Run:  python -m pytest tests -q
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.class_merge import (  # noqa: E402
    RAW_LABEL_KEY, ClassMerge, apply_to_index, class_names, parse_class_merge,
)
from src.metrics import compute_metrics, format_report, resolve_names  # noqa: E402
from src.utils import load_config  # noqa: E402

REPO = Path(__file__).resolve().parent.parent

# Published DAiSEE engagement counts, used as a realistic shape to test against.
RAW = {"train": [34, 213, 2617, 2494],
       "val": [23, 143, 813, 450],
       "test": [4, 84, 882, 814]}


# ----------------------------------------------------------------- test doubles
@dataclass
class FakeRecord:
    split: str
    subject_id: str
    labels: dict = field(default_factory=dict)

    @property
    def engagement(self):
        return self.labels.get("Engagement")


class FakeIndex:
    """Just enough of DatasetIndex for apply_to_index."""

    def __init__(self, counts=RAW):
        self.clips = {
            split: [FakeRecord(split, f"s{i % 7}", {"Engagement": c, "Boredom": 1})
                    for c, n in enumerate(per_class) for i in range(n)]
            for split, per_class in counts.items()
        }

    def all_clips(self):
        for split in ("train", "val", "test"):
            yield from self.clips.get(split, [])

    def counts(self, split):
        out = {}
        for r in self.clips[split]:
            out[r.engagement] = out.get(r.engagement, 0) + 1
        return dict(sorted(out.items()))


def cfg(**dataset):
    return {"dataset": dataset}


# --------------------------------------------------------------------- parsing
def test_absent_class_merge_means_the_original_labelling():
    assert parse_class_merge(cfg(num_classes=4)) is None


def test_partition_is_parsed_and_described():
    m = parse_class_merge(cfg(num_classes=3, class_merge=[[0, 1], [2], [3]]))
    assert m.num_classes == 3
    assert m.mapping == {0: 0, 1: 0, 2: 1, 3: 2}
    assert m.describe()["groups"] == [[0, 1], [2], [3]]


def test_a_dropped_class_is_rejected():
    # [[0, 1], [2]] silently discards level 3; training would then see a label
    # the loss has no output for.
    with pytest.raises(ValueError, match="partition"):
        parse_class_merge(cfg(num_classes=2, class_merge=[[0, 1], [2]]))


def test_a_duplicated_class_is_rejected():
    with pytest.raises(ValueError, match="partition"):
        parse_class_merge(cfg(num_classes=3, class_merge=[[0, 1], [1, 2], [3]]))


def test_an_out_of_range_class_is_rejected():
    with pytest.raises(ValueError, match="outside"):
        parse_class_merge(cfg(num_classes=2, class_merge=[[0, 1], [2, 3, 4]]))


def test_num_classes_disagreeing_with_the_merge_is_an_error():
    # The most dangerous silent failure: tensor shapes come from num_classes.
    with pytest.raises(ValueError, match="num_classes"):
        parse_class_merge(cfg(num_classes=4, class_merge=[[0, 1], [2], [3]]))


def test_binary_merge_is_allowed():
    m = parse_class_merge(cfg(num_classes=2, class_merge=[[0, 1], [2, 3]]))
    assert m.num_classes == 2 and m.mapping == {0: 0, 1: 0, 2: 1, 3: 1}


def test_identity_merge_is_recognised_as_a_no_op():
    m = parse_class_merge(cfg(num_classes=4, class_merge=[[0], [1], [2], [3]]))
    assert m.is_identity()


# ----------------------------------------------------------------------- names
def test_default_names_are_joined_from_the_original_levels():
    m = parse_class_merge(cfg(num_classes=3, class_merge=[[0, 1], [2], [3]]))
    assert m.names == ("Very Low+Low", "High", "Very High")


def test_explicit_names_win():
    c = cfg(num_classes=3, class_merge=[[0, 1], [2], [3]],
            class_names=["Low", "High", "Very High"])
    assert class_names(c) == ("Low", "High", "Very High")


def test_wrong_number_of_explicit_names_is_rejected():
    with pytest.raises(ValueError, match="class_names"):
        parse_class_merge(cfg(num_classes=3, class_merge=[[0, 1], [2], [3]],
                              class_names=["Low", "High"]))


def test_unmerged_config_keeps_the_daisee_level_names():
    assert class_names(cfg(num_classes=4)) == ("Very Low", "Low", "High", "Very High")


# -------------------------------------------------------------- applying it
def test_every_split_is_merged_identically():
    index = FakeIndex()
    merge = parse_class_merge(cfg(num_classes=3, class_merge=[[0, 1], [2], [3]]))
    summary = apply_to_index(index, merge)
    assert index.counts("train") == {0: 247, 1: 2617, 2: 2494}
    assert index.counts("val") == {0: 166, 1: 813, 2: 450}
    assert index.counts("test") == {0: 88, 1: 882, 2: 814}
    assert set(summary["counts"]) == {"train", "val", "test"}


def test_no_clip_is_lost_or_duplicated_by_the_merge():
    index = FakeIndex()
    before = {s: len(v) for s, v in index.clips.items()}
    apply_to_index(index, parse_class_merge(
        cfg(num_classes=3, class_merge=[[0, 1], [2], [3]])))
    assert {s: len(v) for s, v in index.clips.items()} == before
    for split, counts in RAW.items():
        assert sum(index.counts(split).values()) == sum(counts)


def test_the_original_level_is_kept_for_auditing():
    index = FakeIndex({"train": [1, 1, 1, 1]})
    apply_to_index(index, parse_class_merge(
        cfg(num_classes=3, class_merge=[[0, 1], [2], [3]])))
    raws = sorted(r.labels[RAW_LABEL_KEY] for r in index.clips["train"])
    assert raws == [0, 1, 2, 3]


def test_applying_twice_is_idempotent():
    # build_index may be called more than once per process; a second pass must
    # not merge the already-merged values again.
    index = FakeIndex()
    merge = parse_class_merge(cfg(num_classes=3, class_merge=[[0, 1], [2], [3]]))
    apply_to_index(index, merge)
    first = index.counts("train")
    apply_to_index(index, merge)
    assert index.counts("train") == first


def test_other_affect_labels_are_untouched():
    index = FakeIndex({"train": [1, 0, 0, 0]})
    apply_to_index(index, parse_class_merge(
        cfg(num_classes=3, class_merge=[[0, 1], [2], [3]])))
    assert index.clips["train"][0].labels["Boredom"] == 1


def test_identity_merge_changes_nothing():
    index = FakeIndex()
    before = index.counts("train")
    assert apply_to_index(index, parse_class_merge(
        cfg(num_classes=4, class_merge=[[0], [1], [2], [3]]))) is None
    assert index.counts("train") == before


def test_merge_apply_rejects_an_unknown_level():
    m = ClassMerge(groups=((0, 1), (2,), (3,)), names=("a", "b", "c"), source_classes=4)
    with pytest.raises(ValueError, match="not covered"):
        m.apply(7)


# ------------------------------------------------------ the point of the merge
def test_merging_does_not_lower_the_accuracy_baseline():
    """The justification for the merge: the bar does not move.

    Majority-class accuracy on the test split is the same under 4 classes and
    under 3, because "High" is the largest class either way. If this ever
    stopped holding, the merge would be inflating the headline number.
    """
    four = np.array(RAW["test"])
    three = np.array([four[0] + four[1], four[2], four[3]])
    assert four.max() / four.sum() == pytest.approx(three.max() / three.sum())
    assert three.max() / three.sum() == pytest.approx(0.4944, abs=5e-4)


def test_the_merged_class_becomes_measurable():
    # 4 test clips makes a per-class F1 meaningless; 88 does not.
    assert RAW["test"][0] == 4
    assert RAW["test"][0] + RAW["test"][1] == 88


def _majority_baseline(counts):
    y = np.concatenate([np.full(n, c) for c, n in enumerate(counts)])
    pred = np.full_like(y, int(np.argmax(counts)))
    return compute_metrics(y, pred, len(counts))


def test_macro_f1_baseline_does_rise_with_fewer_classes():
    """Honest counterpoint, pinned so the write-up cannot forget it.

    A constant "High" predictor scores a higher macro-F1 on 3 classes than on
    4, because the same single non-zero per-class F1 is divided by 3 instead of
    4. A merged macro-F1 must therefore be read against the merged baseline:

                      4-class   3-class
        val macro-F1   0.1813    0.2417
        test macro-F1  0.1654    0.2206

    Unlike accuracy, this baseline does move, and the exact numbers differ per
    split -- quoting the validation baseline next to a test score would
    understate the bar by 0.076.
    """
    for counts, four_expected, three_expected in (
            ([23, 143, 813, 450], 0.1813, 0.2417),       # val
            ([4, 84, 882, 814], 0.1654, 0.2206),         # test
    ):
        merged = [counts[0] + counts[1], counts[2], counts[3]]
        four = _majority_baseline(counts)["macro_f1"]
        three = _majority_baseline(merged)["macro_f1"]
        assert three > four
        assert four == pytest.approx(four_expected, abs=5e-4)
        assert three == pytest.approx(three_expected, abs=5e-4)


def test_balanced_accuracy_baseline_is_one_over_k_either_way():
    """Why balanced accuracy is the safer selection metric.

    A constant predictor scores exactly 1/K, whatever the prior -- so it cannot
    be won by guessing the class distribution, which is how epoch 0 of the
    4-class run took the best validation accuracy (0.579 against a 0.569
    majority baseline).
    """
    assert _majority_baseline([4, 84, 882, 814])["balanced_accuracy"] == pytest.approx(0.25)
    assert _majority_baseline([88, 882, 814])["balanced_accuracy"] == pytest.approx(1 / 3)


# ----------------------------------------------------------- metric plumbing
def test_metrics_accept_merged_names_and_reject_a_mismatch():
    y = np.array([0, 1, 2, 2])
    m = compute_metrics(y, y, 3, ("Low", "High", "Very High"))
    assert set(m["per_class"]) == {"Low", "High", "Very High"}
    with pytest.raises(ValueError, match="class names"):
        compute_metrics(y, y, 3, ("Low", "High"))


def test_metrics_still_default_to_the_daisee_names():
    m = compute_metrics(np.array([0, 3]), np.array([0, 3]), 4)
    assert "Very High" in m["per_class"]
    assert resolve_names(None, 3) == ("Very Low", "Low", "High")


def test_format_report_uses_the_merged_names():
    y = np.array([0, 1, 2])
    text = format_report(y, y, 3, ("Low", "High", "Very High"))
    assert "Very Low" not in text and "Low" in text


# ------------------------------------------------------------------- config
def test_three_class_config_is_internally_consistent():
    c = load_config(REPO / "configs" / "config_full_3class.yaml")
    m = parse_class_merge(c)
    assert m is not None and m.num_classes == int(c["dataset"]["num_classes"]) == 3
    assert m.groups == ((0, 1), (2), (3)) or m.groups == ((0, 1), (2,), (3,))
    assert class_names(c) == ("Low", "High", "Very High")
    # The changes that make this affordable, asserted as properties rather than
    # as literal numbers, so the reasoning is what is pinned.
    assert int(c["video"]["num_frames"]) == 16

    t = c["training"]
    effective_batch = int(t["batch_size"]) * int(t["grad_accum_steps"])
    # At 1,500 training clips an effective batch of 32 gives only ~1,470
    # optimiser steps over a 32-epoch run, which is what left the layer-decayed
    # rungs unable to move. Accumulation makes a smaller effective batch free:
    # the same clips are processed either way.
    steps = (1500 // effective_batch) * 32
    assert steps >= 2500, f"only {steps} optimiser steps; halve the effective batch"

    # Only the top of the backbone is trained. Driving all twelve blocks at
    # layer-decayed rates meant blocks 0-5 could move at most 2-7% of a weight
    # over the whole run while still costing a full backward pass; measured on
    # a 6 GB card, training four blocks instead is 1.82x faster per clip.
    assert int(c["vit"]["unfreeze_blocks"]) == 4

    # Every trainable rung must be able to move a meaningful fraction of a
    # weight. AdamW's update is about `lr` per step, so lr * steps against a
    # typical 0.03 ViT weight is the bound. The lowest trainable block is
    # finetune_lr * layer_decay**(12 + 1 - 9).
    lowest = float(c["vit"]["finetune_lr"]) * float(c["vit"]["layer_decay"]) ** 4
    assert lowest * steps / 0.03 > 0.5, (
        "the lowest trainable block cannot move half a weight over the run; "
        "either raise finetune_lr or soften layer_decay")
    # Selecting on raw accuracy would have picked the majority-predictor epoch.
    assert c["training"]["selection_metric"] == "balanced_accuracy"
    # T=16 cannot share the T=32 calibration file.
    assert "T16" in c["mrs"]["calibration_file"]


def test_four_class_config_is_left_alone():
    c = load_config(REPO / "configs" / "config_full.yaml")
    assert parse_class_merge(c) is None
    assert int(c["dataset"]["num_classes"]) == 4
    assert int(c["video"]["num_frames"]) == 32
