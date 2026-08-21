"""Step 7a -- STUB. Label-shift early-detection variant.

NOT IMPLEMENTED YET. This module fixes the data structures and function
signatures so the real implementation is a fill-in rather than a redesign.

The idea
--------
Instead of predicting clip t's own engagement label from clip t's features,
predict clip **t+1**'s label from clip **t**'s features, within the same subject
and session. Accuracy is then reported against lead time. Almost nothing in the
engagement literature reports a prediction horizon, so this is the highest-value
result per unit of effort in the project.

What was found about clip ordering
----------------------------------
DAiSEE ships no explicit session or ordering metadata, so ordering has to come
from the ClipID. Inspecting all 108 sample clips shows a consistent encoding:

    ClipID = <SubjectID: 6 digits><suffix>

with the suffix being 4 digits in 106 of 108 cases and 3 in the other two
(``826412010``, ``556463012`` -- these read as a stripped leading zero). Padding
the suffix left to 4 digits gives:

    session index = suffix[0]        observed values 0, 1, 2
    clip index    = suffix[1:]       observed range 001-240

Evidence this is a real ordering and not a coincidence:
  * every subject's clips share a small number of session digits (e.g. subject
    110001 has ten clips in session 1 and one in session 2);
  * clip indices are strictly increasing integers within a session;
  * DAiSEE's collection protocol cut fixed 10-second segments from longer
    recordings, which is consistent with a monotone within-session index.

IMPORTANT CAVEAT, to be restated in the writeup: this is *inferred* from the ID
pattern, not documented by the dataset authors. And indices are not dense -- the
sample's subject 110001 has 1002, 1003, ..., 1012, 1040, 1048 -- so consecutive
*indices* do not always mean temporally adjacent clips. Any real implementation
must therefore only pair clips whose indices differ by exactly 1 and record the
implied lead time, rather than assuming every neighbouring pair is 10 s apart.

Nothing below is called by train.py yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

SUFFIX_WIDTH = 4
SUBJECT_ID_WIDTH = 6


@dataclass(frozen=True)
class ClipOrder:
    """Position of a clip within its subject's recording sessions."""

    subject_id: str
    clip_id: str
    session: int
    index: int
    parsed_cleanly: bool
    """False when the ClipID did not match the expected pattern and the values
    above are a best guess -- such clips must be excluded from pairing."""


@dataclass(frozen=True)
class ShiftedPair:
    """One (source clip -> target clip) training pair for early detection."""

    source_clip_id: str
    target_clip_id: str
    subject_id: str
    split: str
    index_gap: int
    """Difference in clip index. 1 == immediately consecutive."""
    lead_time_s: float
    """Nominal seconds ahead, = index_gap * clip_duration_s."""
    target_label: int


def parse_clip_order(clip_id: str, subject_id: str) -> ClipOrder:
    """Decompose a ClipID into (session, index). SEE MODULE CAVEAT.

    TODO: implement. Strip the subject prefix, left-pad the remainder to
    SUFFIX_WIDTH with zeros, then split into session digit and 3-digit index.
    Set ``parsed_cleanly=False`` when the ClipID does not start with the subject
    ID or the suffix is not all digits.
    """
    raise NotImplementedError("parse_clip_order is a stub; see module docstring.")


def build_shifted_pairs(
    clip_metadata: Sequence[Dict[str, object]],
    max_index_gap: int = 1,
    clip_duration_s: float = 10.0,
    target: str = "Engagement",
) -> List[ShiftedPair]:
    """Pair each clip with a later clip from the same subject AND session.

    TODO: implement. Group by (subject_id, session), sort by index, and emit a
    pair for every (clip_i, clip_j) whose index gap is <= max_index_gap. Never
    pair across sessions, and never pair across splits (which cannot happen
    anyway while DAiSEE stays subject-disjoint, but assert it).
    """
    raise NotImplementedError("build_shifted_pairs is a stub; see module docstring.")


def build_early_detection_dataset(
    output_root: Path,
    splits: Sequence[str],
    max_index_gap: int = 1,
    affect_feature_set: str = "probs",
):
    """Return a Dataset yielding (features of clip t, label of clip t+1).

    TODO: implement by reusing ``datasets.load_clip_samples`` for the features
    and re-keying the labels through ``build_shifted_pairs``. The existing
    ``EngagementClipDataset`` can be subclassed -- only the label lookup changes,
    so the models and training loop need no modification at all.
    """
    raise NotImplementedError("build_early_detection_dataset is a stub.")


def evaluate_by_lead_time(
    predictions: Sequence[int],
    pairs: Sequence[ShiftedPair],
    num_classes: int = 4,
) -> Dict[float, Dict[str, float]]:
    """Group metrics by lead time -- the headline table for this experiment.

    TODO: implement. Bucket by ``pair.lead_time_s``, call ``metrics.evaluate``
    per bucket, and return ``{lead_time: metric_bundle}`` so the writeup can plot
    macro-F1 against prediction horizon.
    """
    raise NotImplementedError("evaluate_by_lead_time is a stub.")
