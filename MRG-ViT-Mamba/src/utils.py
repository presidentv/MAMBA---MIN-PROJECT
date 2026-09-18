"""Shared helpers: config loading, seeding, device selection, logging, JSON I/O.

Nothing in here makes assumptions about the dataset or the models. Anything that
must be *discovered* (CUDA presence, package versions, tensor dimensions) is
queried at runtime and returned as data, never hard-coded.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
class Config(dict):
    """dict with attribute access and dotted-path lookup, so cfg.video.num_frames
    and cfg.dotted("paths.cache_dir") both work."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:  # pragma: no cover - programmer error
            raise AttributeError(item) from exc
        return Config(value) if isinstance(value, dict) else value

    def dotted(self, path: str, default: Any = None) -> Any:
        node: Any = self
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


# Environment variables that override config paths, so the same committed
# config runs on a machine whose dataset, cache or checkpoint locations differ
# from this one's without anyone editing a tracked file.
PATH_ENV_OVERRIDES = {
    "DAISEE_ROOT": "dataset_root",
    "MRG_CACHE_DIR": "cache_dir",
    "MRG_CHECKPOINT_DIR": "checkpoints_dir",
}


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else REPO_ROOT / "configs" / "config.yaml"
    if not path.is_absolute() and not path.exists():
        path = REPO_ROOT / path
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    applied = {}
    for env, key in PATH_ENV_OVERRIDES.items():
        value = os.environ.get(env)
        if value:
            raw.setdefault("paths", {})[key] = value
            applied[key] = f"{env}={value}"
    # Recorded in the config itself so every artifact written from it shows
    # where the data actually came from.
    raw["_path_overrides"] = applied
    return Config(raw)


def missing_clip_policy(cfg) -> tuple[bool, float]:
    """(allow_missing, max_fraction) for clips that could not be processed.

    The balanced development subsets are small enough that one failed clip means
    something is wrong, so the default is to stop. The full DAiSEE release has a
    handful of videos that do not decode; stopping a many-hour run for them
    would be worse than skipping them. ``dataset.missing_clip_policy: skip``
    allows that, but only up to ``dataset.max_missing_fraction`` of a split --
    beyond it the cause is almost certainly an incomplete preprocessing run,
    not bad videos, and training on what is left would be silently wrong.
    """
    ds = cfg["dataset"] if "dataset" in cfg else {}
    policy = str(ds.get("missing_clip_policy", "error")).lower()
    if policy not in ("error", "skip"):
        raise ValueError(f"dataset.missing_clip_policy must be 'error' or 'skip', got {policy!r}")
    frac = float(ds.get("max_missing_fraction", 0.0)) if policy == "skip" else 0.0
    if not 0.0 <= frac < 1.0:
        raise ValueError(f"dataset.max_missing_fraction must be in [0, 1), got {frac}")
    return policy == "skip", frac


def resolve_path(value: str | Path) -> Path:
    """Resolve a config path relative to the repository root."""
    p = Path(value)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def ensure_dir(path: str | Path) -> Path:
    p = resolve_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover
        pass


# --------------------------------------------------------------------------- #
# Device
# --------------------------------------------------------------------------- #
@dataclass
class DeviceInfo:
    device: str
    name: str
    total_memory_bytes: int | None
    cuda_available: bool
    torch_version: str
    torch_cuda_version: str | None

    def as_dict(self) -> dict:
        return {
            "device": self.device,
            "name": self.name,
            "total_memory_bytes": self.total_memory_bytes,
            "total_memory_gb": (
                round(self.total_memory_bytes / 1024**3, 2)
                if self.total_memory_bytes
                else None
            ),
            "cuda_available": self.cuda_available,
            "torch_version": self.torch_version,
            "torch_cuda_version": self.torch_cuda_version,
        }


def get_device(prefer: str = "auto") -> DeviceInfo:
    """Query the *actual* hardware. Never assumes CUDA exists (spec section 25)."""
    import torch

    cuda = torch.cuda.is_available()
    if prefer == "cpu" or not cuda:
        return DeviceInfo(
            device="cpu",
            name=platform.processor() or "cpu",
            total_memory_bytes=None,
            cuda_available=cuda,
            torch_version=torch.__version__,
            torch_cuda_version=torch.version.cuda,
        )
    props = torch.cuda.get_device_properties(0)
    return DeviceInfo(
        device="cuda",
        name=torch.cuda.get_device_name(0),
        total_memory_bytes=props.total_memory,
        cuda_available=True,
        torch_version=torch.__version__,
        torch_cuda_version=torch.version.cuda,
    )


def gpu_memory_mb() -> float | None:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024**2
    except Exception:
        pass
    return None


# --------------------------------------------------------------------------- #
# Logging / JSON
# --------------------------------------------------------------------------- #
def get_logger(name: str, log_file: str | Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file:
        path = resolve_path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    return logger


class _NpEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def save_json(obj: Any, path: str | Path) -> Path:
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, cls=_NpEncoder)
    return p


def load_json(path: str | Path) -> Any:
    with open(resolve_path(path), "r", encoding="utf-8") as fh:
        return json.load(fh)


# --------------------------------------------------------------------------- #
# Environment record
# --------------------------------------------------------------------------- #
def package_version(name: str) -> str | None:
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return None


def git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def environment_record() -> dict:
    """Everything needed to reproduce a run. Written next to every result file."""
    info: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": git_commit(),
        "packages": {
            name: package_version(name)
            for name in [
                "torch", "torchvision", "timm", "cv2", "mediapipe",
                "numpy", "sklearn", "shap", "pandas", "matplotlib", "einops", "yaml",
            ]
        },
    }
    try:
        info["device"] = get_device().as_dict()
    except Exception as exc:  # torch missing
        info["device"] = {"error": repr(exc)}
    return info


class Timer:
    """Context manager returning wall-clock seconds in .elapsed."""

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.start
        return False
