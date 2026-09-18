"""Download the MediaPipe Tasks model bundles into models/.

They are not committed to the repository: they are third-party binaries with
their own licence, and they are reproducibly fetchable from Google's published
model URLs.
"""

from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.face_processing import FACE_DETECTOR_URL  # noqa: E402
from src.landmarks import FACE_LANDMARKER_URL  # noqa: E402
from src.utils import ensure_dir, load_config, save_json  # noqa: E402

BUNDLES = {
    "blaze_face_short_range.tflite": FACE_DETECTOR_URL,
    "face_landmarker.task": FACE_LANDMARKER_URL,
}


def main() -> int:
    cfg = load_config()
    out_dir = ensure_dir(cfg.paths["models_dir"])
    record = {}
    for name, url in BUNDLES.items():
        dest = out_dir / name
        if dest.exists():
            print(f"[skip] {name} already present ({dest.stat().st_size} bytes)")
        else:
            print(f"[get ] {name} <- {url}")
            try:
                urllib.request.urlretrieve(url, dest)
            except Exception as exc:
                print(f"FAILED to download {name}: {type(exc).__name__}: {exc}")
                return 1
            print(f"       saved {dest.stat().st_size} bytes")
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()
        record[name] = {"url": url, "path": str(dest),
                        "bytes": dest.stat().st_size, "sha256": digest}
        print(f"       sha256 {digest}")

    save_json({"bundles": record}, "artifacts/model_bundles.json")
    print("\nwrote artifacts/model_bundles.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
