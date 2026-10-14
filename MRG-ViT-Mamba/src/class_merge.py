"""Collapsing DAiSEE's four engagement levels into fewer classes.

Why this exists. The engagement label is skewed past the point where the rarest
level can be learned or even measured:

    level      train   val   test
    0 Very Low    34    23      4
    1 Low        213   143     84
    2 High     2,617   813    882
    3 Very High 2,494   450    814

Four test clips means one clip is 25% of that class's recall, so its F1 is noise
no matter how training goes. Merging levels 0 and 1 gives a class with 247
training and 88 test clips, which is measurable.

The merge is worth doing because it does *not* move the bar. Majority-class
accuracy on the test split is 882/1784 = 0.494 under both the 4-class and the
3-class labelling, because "High" is the largest class either way. So nothing is
inflated: the class that cannot be learned is removed, and the number to beat
stays the same.

What this module guarantees. The remapping is applied in one place --
``build_index`` -- which is the single point where a label enters the pipeline.
Training, sampling, loss weighting, metrics, confusion matrices and the test
evaluation therefore all see the same labelling, and it is impossible for the
train split to be merged while the test split is not. The original value is
retained as ``Engagement_raw`` on every record so the mapping stays auditable.

Nothing in the Stage 1 or Stage 2 cache keys depends on labels (see
``preprocess.stage1_key``), so switching the merge on or off re-uses the
existing cache and never triggers reprocessing.

Config::

    dataset:
      num_classes: 3
      class_merge: [[0, 1], [2], [3]]
      class_names: ["Low", "High", "Very High"]   # optional

``num_classes`` must equal the number of groups; a disagreement is an error
rather than something resolved silently, because every downstream tensor shape
is built from it.
"""

from __future__ import annotations

from dataclasses import dataclass

from .metrics import DEFAULT_CLASS_NAMES

RAW_LABEL_KEY = "Engagement_raw"


@dataclass(frozen=True)
class ClassMerge:
    """A validated partition of the original classes into merged groups."""

    groups: tuple[tuple[int, ...], ...]
    names: tuple[str, ...]
    source_classes: int

    @property
    def num_classes(self) -> int:
        return len(self.groups)

    @property
    def mapping(self) -> dict[int, int]:
        return {raw: new for new, group in enumerate(self.groups) for raw in group}

    def apply(self, raw: int) -> int:
        try:
            return self.mapping[int(raw)]
        except KeyError as exc:
            raise ValueError(
                f"engagement label {raw!r} is not covered by class_merge "
                f"{[list(g) for g in self.groups]}") from exc

    def is_identity(self) -> bool:
        return all(len(g) == 1 for g in self.groups)

    def describe(self) -> dict:
        return {
            "groups": [list(g) for g in self.groups],
            "names": list(self.names),
            "mapping": {str(k): v for k, v in sorted(self.mapping.items())},
            "source_classes": self.source_classes,
            "num_classes": self.num_classes,
        }


def _default_name(group: tuple[int, ...], source_names: tuple[str, ...]) -> str:
    parts = [source_names[i] if i < len(source_names) else str(i) for i in group]
    return parts[0] if len(parts) == 1 else "+".join(parts)


def parse_class_merge(cfg, source_classes: int = 4,
                      source_names: tuple[str, ...] = DEFAULT_CLASS_NAMES
                      ) -> ClassMerge | None:
    """Read ``dataset.class_merge``; None when absent (the original labelling).

    Validation is strict on purpose. A partition that drops a class would train
    on fewer labels than the data contains and fail only later, as an index
    error inside the loss, or not at all.
    """
    ds = cfg["dataset"] if "dataset" in cfg else {}
    raw = ds.get("class_merge")
    if raw in (None, False):
        return None
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError(f"dataset.class_merge must be a non-empty list of lists, got {raw!r}")

    groups: list[tuple[int, ...]] = []
    seen: list[int] = []
    for entry in raw:
        if isinstance(entry, int):
            entry = [entry]
        if not isinstance(entry, (list, tuple)) or not entry:
            raise ValueError(f"each class_merge group must be a non-empty list, got {entry!r}")
        group = tuple(int(x) for x in entry)
        for x in group:
            if not 0 <= x < source_classes:
                raise ValueError(
                    f"class_merge refers to class {x}, outside 0..{source_classes - 1}")
        groups.append(group)
        seen.extend(group)

    if sorted(seen) != list(range(source_classes)):
        missing = sorted(set(range(source_classes)) - set(seen))
        duplicated = sorted({x for x in seen if seen.count(x) > 1})
        raise ValueError(
            f"class_merge must partition all {source_classes} classes exactly once; "
            f"missing={missing} duplicated={duplicated}")

    explicit = ds.get("class_names")
    if explicit is not None:
        names = tuple(str(n) for n in explicit)
        if len(names) != len(groups):
            raise ValueError(
                f"dataset.class_names has {len(names)} entries but class_merge defines "
                f"{len(groups)} classes")
    else:
        names = tuple(_default_name(g, source_names) for g in groups)

    declared = ds.get("num_classes")
    if declared is not None and int(declared) != len(groups):
        raise ValueError(
            f"dataset.num_classes is {declared} but class_merge defines {len(groups)} "
            f"classes. Set num_classes to {len(groups)}.")

    return ClassMerge(groups=tuple(groups), names=names, source_classes=source_classes)


def class_names(cfg, num_classes: int | None = None) -> tuple[str, ...]:
    """Display names for the labelling this config describes.

    Used by every report, classification table and confusion-matrix axis, so a
    merged run is never labelled with the four original level names.
    """
    merge = parse_class_merge(cfg)
    if merge is not None:
        return merge.names
    ds = cfg["dataset"] if "dataset" in cfg else {}
    explicit = ds.get("class_names")
    if explicit is not None:
        return tuple(str(n) for n in explicit)
    n = int(num_classes if num_classes is not None else ds.get("num_classes", 4))
    return DEFAULT_CLASS_NAMES[:n] if n <= len(DEFAULT_CLASS_NAMES) else tuple(
        f"class {i}" for i in range(n))


def apply_to_index(index, merge: ClassMerge | None, target_label: str = "Engagement"):
    """Rewrite every record's label in place, keeping the original alongside.

    Returns a summary mapping split -> {merged_class: count} so the caller can
    log what the labelling actually became rather than what it intended.
    """
    if merge is None or merge.is_identity():
        return None
    summary: dict[str, dict[int, int]] = {}
    for record in index.all_clips():
        raw = record.labels.get(target_label)
        if raw is None:
            continue
        if RAW_LABEL_KEY not in record.labels:
            record.labels[RAW_LABEL_KEY] = int(raw)
        new = merge.apply(record.labels[RAW_LABEL_KEY])
        record.labels[target_label] = new
        bucket = summary.setdefault(record.split, {})
        bucket[new] = bucket.get(new, 0) + 1
    return {"merge": merge.describe(),
            "counts": {s: dict(sorted(c.items())) for s, c in sorted(summary.items())}}
