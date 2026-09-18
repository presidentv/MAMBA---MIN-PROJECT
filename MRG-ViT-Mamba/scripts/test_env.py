"""TEST 1 - environment.

Records the actual versions of everything the pipeline depends on, the actual
device, and which Mamba backend is in use. Writes artifacts/environment.json.
Exits non-zero if a hard requirement is missing.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import environment_record, get_device, package_version, save_json  # noqa: E402

REQUIRED = ["torch", "torchvision", "timm", "cv2", "mediapipe", "numpy", "sklearn",
            "shap", "pandas", "matplotlib", "einops", "yaml"]
OPTIONAL = ["transformers", "mamba_ssm"]


def main() -> int:
    print("=" * 72)
    print("TEST 1 - ENVIRONMENT")
    print("=" * 72)
    print(f"Python   : {sys.version.split()[0]}")

    missing = []
    for name in REQUIRED:
        v = package_version(name)
        print(f"{name:<14}: {v if v else 'MISSING'}")
        if v is None:
            missing.append(name)
    for name in OPTIONAL:
        v = package_version(name)
        print(f"{name:<14}: {v if v else 'not installed (optional)'}")

    dev = get_device()
    print("\n--- device ---")
    for k, v in dev.as_dict().items():
        print(f"{k:<20}: {v}")

    print("\n--- mamba backend ---")
    from src.mamba_ref import mamba_backend_info

    backend = mamba_backend_info()
    for k, v in backend.items():
        print(f"{k:<26}: {v}")

    record = environment_record()
    record["mamba_backend"] = backend
    record["missing_required"] = missing
    save_json(record, "artifacts/environment.json")
    print("\nwrote artifacts/environment.json")

    if missing:
        print(f"\nTEST 1 FAILED - missing required packages: {missing}")
        return 1
    print("\nTEST 1 PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
