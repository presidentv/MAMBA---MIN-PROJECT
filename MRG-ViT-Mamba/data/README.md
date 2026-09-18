# Dataset setup

This project uses **DAiSEE**. The dataset is not included in this repository and
must not be redistributed: access requires per-user registration, and the clips
are video of identifiable people.

* Project page: https://people.iith.ac.in/vineethnb/resources/daisee/
* Paper: https://arxiv.org/abs/1609.01885

## Expected layout

Point `dataset.root` in `configs/config.yaml` at a directory laid out the way
DAiSEE ships (this is the structure described in the dataset's own README.txt):

```
<DAISEE_ROOT>/
├── DataSet/
│   ├── Train/<SubjectID>/<ClipID>/<ClipID>.avi
│   ├── Validation/<SubjectID>/<ClipID>/<ClipID>.avi
│   └── Test/<SubjectID>/<ClipID>/<ClipID>.avi
└── Labels/
    ├── TrainLabels.csv
    ├── ValidationLabels.csv
    └── TestLabels.csv
```

Label CSVs have the columns `ClipID,Boredom,Engagement,Confusion,Frustration`.
Only **Engagement** is used. The subject id is the directory one level above the
clip directory; that is what makes the subject-disjointness check possible.

## Verify before doing anything else

```bash
python scripts/audit_dataset.py --probe-all
```

This resolves every clip, joins it to its label row, decode-probes the videos,
and checks that no subject appears in more than one split. It exits non-zero and
refuses to continue if anything is wrong. Nothing downstream should be run until
it passes.

## Notes on partial copies

The code does not assume a clip count. If you have a subset, the audit reports
exactly what it found (including label rows with no matching video file) and the
pipeline runs on what is there. Any results are then results *on that subset* and
must be reported as such -- they are not comparable to published DAiSEE numbers.
