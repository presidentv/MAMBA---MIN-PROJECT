# Front end

A local web page that scores a ten-second webcam clip with the fine-tuned
binary detector and shows why it decided what it did.

## Running it

```
.venv/Scripts/python.exe app/server.py --checkpoint <path-to>/fullbin/best.pt
```

Then open <http://127.0.0.1:8000/>. The model takes a few seconds to load; once
it is up, a clip is scored in roughly one to three seconds.

The checkpoint is not in this repository — it is 333 MB. The run that produced
it records where it went, in `artifacts/finetune_report_fullbin.json` under
`checkpoint_dir`.

## Clips

Any video dropped into `app/demo/` is listed on the page. That directory is
git-ignored: the clips are webcam recordings of identifiable people and have no
business in version control. A clip can also be uploaded from the page, which
scores it and deletes it again.

## What is on screen

* **The gauge** is P(disengaged) — the quantity the paper's ROC-AUC ranks on,
  not a verdict. At this operating point the detector finds 54 of 88 disengaged
  test clips and raises 255 false alarms, so a "not engaged" call is correct
  roughly one time in six. The page says so, under the cards. Anyone reading
  the results table will check this, and a page that implied more would not
  survive the question.

* **The reliability strip** is the per-frame Motion Reliability Score, the
  contribution this project is actually about. All sixteen frames are present
  in every strip; the short bars are frames the model trusted less. Nothing is
  dropped, which is the whole argument.

* **What the decision rests on** is integrated gradients over the named
  landmark features, towards the predicted class.

## Why the standard library

`server.py` uses `http.server`, not FastAPI. One local user needs nothing more,
and this environment is pinned around torch 2.11, timm 1.0.29 and a reference
Mamba; installing a web stack into it hours before a demonstration is a poor
trade. Uploads therefore arrive as a raw request body with the name in a
header, because the standard library's multipart parser was removed in Python
3.13.

The server binds to loopback by default. `--host 0.0.0.0` would put webcam
video of real people on the local network, so it is not the default.

## Checking it still matches the evaluated model

`src/inference.py` builds the model input itself, so it can drift from the path
the reported metrics came from without anything raising. After changing the
preprocessing, the MRS, the fusion scaler or the evaluation protocol:

```
.venv/Scripts/python.exe scripts/check_inference.py \
    --checkpoint <path-to>/fullbin/best.pt \
    --cache-dir  <cache>/faces_T16_<key>/test
```

It compares against `artifacts/per_clip_predictions_fullbin_test.json`. Add
`--from-video --video-dir <DAiSEE>` when the dataset drive is attached to cover
face detection and the landmark mesh as well.
