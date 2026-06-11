"""Shared constants: cache locations, license policy, well-known class lists."""

from __future__ import annotations

import os
from pathlib import Path

# --- cache / environment ---------------------------------------------------

#: Environment variable pointing at the ovkit cache root.
ENV_HOME = "OVKIT_HOME"
#: Environment variable; when ``"1"`` no network access is attempted.
ENV_OFFLINE = "OVKIT_OFFLINE"


def cache_root() -> Path:
    """Return the ovkit cache root, honoring ``$OVKIT_HOME``.

    Defaults to ``~/.cache/ovkit``. The directory is created on demand.
    """
    root = os.environ.get(ENV_HOME)
    base = Path(root).expanduser() if root else Path.home() / ".cache" / "ovkit"
    return base


def is_offline() -> bool:
    """Return ``True`` when offline mode is requested via ``$OVKIT_OFFLINE``."""
    return os.environ.get(ENV_OFFLINE, "").strip() in {"1", "true", "True", "yes"}


# --- license policy --------------------------------------------------------

#: SPDX ids accepted for models registered in the manifest. Anything outside
#: this set (AGPL, non-commercial weights, ...) must not ship with ovkit.
PERMISSIVE_LICENSES: frozenset[str] = frozenset(
    {
        "apache-2.0",
        "mit",
        "bsd-2-clause",
        "bsd-3-clause",
        "bsd",
        "isc",
        "unlicense",
        "cc0-1.0",
        "mpl-2.0",
    }
)


def is_permissive(license_id: str | None) -> bool:
    """Return ``True`` if ``license_id`` is a known permissive SPDX id."""
    if not license_id:
        return False
    return license_id.strip().lower() in PERMISSIVE_LICENSES


# --- class name tables -----------------------------------------------------

#: 80 COCO class names (detection/segmentation), indexed by class id.
COCO80: tuple[str, ...] = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)

#: Registry of named class tables referenced from the manifest ``classes`` key.
CLASS_TABLES: dict[str, tuple[str, ...]] = {
    "coco80": COCO80,
}


def class_names(key: str | None, num_classes: int | None = None) -> dict[int, str]:
    """Resolve a manifest ``classes`` key to an ``{id: name}`` mapping.

    Falls back to ``class_<i>`` names when ``key`` is unknown. ``num_classes``,
    when given, sizes the fallback table.
    """
    if key and key in CLASS_TABLES:
        return {i: n for i, n in enumerate(CLASS_TABLES[key])}
    n = num_classes if num_classes is not None else 0
    return {i: f"class_{i}" for i in range(n)}
