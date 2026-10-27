"""Local web front end for the disengagement detector.

Run it, open the page, pick a clip, see what the model says and why.

    python app/server.py --checkpoint C:/Users/ashok/mrg_run/checkpoints/fullbin/best.pt

Design notes:

* The standard library serves this, not FastAPI. One local user needs nothing
  more, and the environment this model runs in is pinned (torch 2.11, timm
  1.0.29, a reference Mamba); installing a web stack into it before a live
  demonstration risks far more than it buys. Uploads therefore arrive as a raw
  body with the name in a header rather than as multipart, because the
  stdlib's multipart parser (cgi) was removed in Python 3.13.

* The model is loaded once at startup and guarded by a lock. Loading takes a
  few seconds and ~1 GB of VRAM; doing it per request would make the page feel
  broken and would eventually exhaust the 6 GB card.

* Nothing leaves the machine. The page is bound to the loopback interface by
  default, and clips are read from disk and scored in place. That is not
  incidental: webcam video of identifiable people is exactly the data this
  project has kept off the network throughout.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import re
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inference import EngagementPredictor  # noqa: E402

WEB = Path(__file__).resolve().parent / "web"
DEMO = Path(__file__).resolve().parent / "demo"
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_UPLOAD = 256 * 1024 * 1024

# Reported in the paper for this checkpoint at the default threshold. The page
# shows these so a viewer can judge a single prediction against the model's
# measured behaviour instead of trusting the badge.
TEST_METRICS = {
    "roc_auc": 0.7943,
    "average_precision": 0.2628,
    "balanced_accuracy": 0.7316,
    "accuracy": 0.8380,
    "recall_not_engaged": 0.614,
    "precision_not_engaged": 0.175,
    "found": 54,
    "total_not_engaged": 88,
    "false_alarms": 255,
    "test_clips": 1784,
    "constant_engaged_accuracy": 0.9507,
}

_state = {"predictor": None, "lock": threading.Lock()}


def clip_id(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", path.stem).strip("-").lower()


def demo_clips() -> dict[str, Path]:
    if not DEMO.is_dir():
        return {}
    found = {}
    for p in sorted(DEMO.iterdir()):
        if p.suffix.lower() in VIDEO_SUFFIXES:
            found[clip_id(p)] = p
    return found


def predict(path: Path, explain: bool = True) -> dict:
    """Score one clip. Serialised: one model, one GPU, one request at a time."""
    predictor = _state["predictor"]
    started = time.time()
    with _state["lock"]:
        result = predictor.predict(path, explain=explain)
    payload = result.as_dict()
    payload["elapsed_s"] = round(time.time() - started, 2)
    payload["name"] = path.name
    return payload


class Handler(BaseHTTPRequestHandler):
    server_version = "MRGViTMamba/1.0"
    protocol_version = "HTTP/1.1"

    # ------------------------------------------------------------ plumbing
    def log_message(self, fmt, *args):
        sys.stderr.write(f"  {self.address_string()} {fmt % args}\n")

    def _send(self, code, body=b"", ctype="application/octet-stream", extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        # The page reads only its own origin; no caching, so a rebuilt asset is
        # never served stale during a demonstration.
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD" and body:
            self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode("utf-8"), "application/json")

    def _error(self, code, message):
        self._json({"error": message}, code)

    def _serve_file(self, path: Path):
        if not path.is_file():
            return self._error(404, f"not found: {path.name}")
        ctype = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self._send(200, path.read_bytes(), ctype)

    def _serve_video(self, path: Path):
        """Serve a video, honouring Range so the browser can seek.

        Chrome requests a byte range for <video> and will not scrub a response
        that ignores it, so this answers 206 with the requested slice.
        """
        if not path.is_file():
            return self._error(404, "clip not found")
        size = path.stat().st_size
        ctype = mimetypes.guess_type(str(path))[0] or "video/mp4"
        rng = self.headers.get("Range")
        m = re.match(r"bytes=(\d*)-(\d*)", rng or "")
        if not m:
            with open(path, "rb") as fh:
                data = fh.read()
            return self._send(200, data, ctype, {"Accept-Ranges": "bytes"})
        start = int(m.group(1)) if m.group(1) else 0
        end = int(m.group(2)) if m.group(2) else size - 1
        end = min(end, size - 1)
        if start > end:
            return self._send(416, b"", ctype, {"Content-Range": f"bytes */{size}"})
        with open(path, "rb") as fh:
            fh.seek(start)
            data = fh.read(end - start + 1)
        self._send(206, data, ctype, {
            "Accept-Ranges": "bytes",
            "Content-Range": f"bytes {start}-{end}/{size}",
        })

    # ----------------------------------------------------------------- GET
    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        try:
            route = self.path.split("?")[0].rstrip("/") or "/"
            if route in ("/", "/index.html"):
                return self._serve_file(WEB / "index.html")
            if route.startswith("/static/"):
                name = route[len("/static/"):]
                target = (WEB / name).resolve()
                # keep the handler inside the web directory
                if not str(target).startswith(str(WEB.resolve())):
                    return self._error(403, "forbidden")
                return self._serve_file(target)
            if route == "/api/model":
                info = dict(_state["predictor"].describe())
                info["test_metrics"] = TEST_METRICS
                return self._json(info)
            if route == "/api/clips":
                return self._json({"clips": [
                    {"id": cid, "name": p.name,
                     "size_mb": round(p.stat().st_size / 1e6, 1)}
                    for cid, p in demo_clips().items()]})
            if route.startswith("/api/clip/"):
                cid = route[len("/api/clip/"):]
                clips = demo_clips()
                if cid not in clips:
                    return self._error(404, "clip not found")
                return self._serve_video(clips[cid])
            return self._error(404, "no such route")
        except Exception:
            traceback.print_exc()
            self._error(500, "internal error; see the server log")

    # ---------------------------------------------------------------- POST
    def do_POST(self):
        try:
            route = self.path.split("?")[0].rstrip("/")
            if route == "/api/predict":
                return self._predict_demo()
            if route == "/api/upload":
                return self._predict_upload()
            return self._error(404, "no such route")
        except Exception:
            traceback.print_exc()
            self._error(500, "internal error; see the server log")

    def _body(self) -> bytes:
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_UPLOAD:
            raise ValueError(f"body of {n} bytes exceeds the {MAX_UPLOAD} limit")
        return self.rfile.read(n)

    def _predict_demo(self):
        req = json.loads(self._body() or b"{}")
        clips = demo_clips()
        cid = req.get("clip")
        if cid not in clips:
            return self._error(404, f"unknown clip {cid!r}")
        return self._json(predict(clips[cid], explain=bool(req.get("explain", True))))

    def _predict_upload(self):
        """Score an uploaded clip.

        The body is the raw file. Multipart would need a parser the standard
        library no longer ships, and this is a local page talking to its own
        server, so the simpler contract is the better one.
        """
        name = self.headers.get("X-Filename") or "upload.mp4"
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", os.path.basename(name))
        if Path(name).suffix.lower() not in VIDEO_SUFFIXES:
            return self._error(400, f"{name}: not a recognised video file")
        data = self._body()
        if not data:
            return self._error(400, "empty upload")
        tmp_dir = DEMO / "_uploads"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp = tmp_dir / f"{int(time.time()*1000)}_{name}"
        tmp.write_bytes(data)
        try:
            return self._json(predict(tmp, explain=True))
        finally:
            # The upload is someone's face; it is scored and removed, not kept.
            try:
                tmp.unlink()
            except OSError:
                pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", required=True, help="a fine-tuned best.pt")
    ap.add_argument("--host", default="127.0.0.1",
                    help="loopback by default; the clips are of real people")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-tta", action="store_true",
                    help="skip the evaluated three-view averaging (faster, "
                         "but no longer the protocol the paper reports)")
    args = ap.parse_args(argv)

    print("loading the model ...")
    t0 = time.time()
    _state["predictor"] = EngagementPredictor.from_checkpoint(
        args.checkpoint, device=args.device, use_tta=not args.no_tta)
    print(f"ready in {time.time() - t0:.1f}s")
    print(json.dumps(_state["predictor"].describe(), indent=2))

    clips = demo_clips()
    if clips:
        print(f"\ndemo clips in {DEMO}:")
        for cid, p in clips.items():
            print(f"   {cid:24s} {p.name}")
    else:
        print(f"\nno demo clips found. Drop video files into {DEMO} "
              f"to have them listed on the page.")

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"\n  open {url}\n  ctrl-c to stop\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        httpd.server_close()
        _state["predictor"].close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
