"""Shared constants: cache locations, license policy, well-known class lists."""

from __future__ import annotations

import os
from functools import cache
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

#: 21 Pascal VOC classes (index 0 is background), for VOC-trained segmenters.
VOC21: tuple[str, ...] = (
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)

#: 53 environmental sound classes recognised by AclNet.
ACLNET53: tuple[str, ...] = (
    "dog",
    "rooster",
    "pig",
    "cow",
    "frog",
    "cat",
    "hen",
    "insects (flying)",
    "sheep",
    "crow",
    "rain",
    "sea waves",
    "crackling fire",
    "crickets",
    "chirping birds",
    "water drops",
    "wind",
    "pouring water",
    "toilet flush",
    "thunderstorm",
    "crying baby",
    "sneezing",
    "clapping",
    "breathing",
    "coughing",
    "footsteps",
    "laughing",
    "brushing teeth",
    "snoring",
    "drinking sipping",
    "door knock",
    "mouse click",
    "keyboard typing",
    "door wood creaks",
    "can opening",
    "washing machine",
    "vacuum cleaner",
    "clock alarm",
    "clock tick",
    "glass breaking",
    "helicopter",
    "chainsaw",
    "siren",
    "car horn",
    "engine",
    "train",
    "church bells",
    "airplane",
    "fireworks",
    "hand saw",
    "gunshot",
    "crowd",
    "speech",
)

#: Registry of named class tables referenced from the manifest ``classes`` key.
#:
#: Every table below is transcribed from the model's own documented interface
#: (its Open Model Zoo README or dataset class file) — a model that answers
#: "class_3" has not answered at all, so each mirrored model that ships no
#: ``labels.txt`` gets its table wired up in ``manifests/labels.yaml``.
CLASS_TABLES: dict[str, tuple[str, ...]] = {
    "coco80": COCO80,
    "voc21": VOC21,
    "aclnet53": ACLNET53,
    # Single-class detectors: the class id carries no information, the model name does.
    "face": ("face",),
    "person": ("person",),
    "vehicle": ("vehicle",),
    "text": ("text",),
    # person-vehicle-bike-detection-2000: labels 0/1/2.
    "person_vehicle_bike": ("vehicle", "person", "bike"),
    # vehicle-license-plate-detection-barrier-0106: labels 1 and 2 (0 unused).
    "vehicle_plate": ("background", "vehicle", "license plate"),
    # product-detection-0001: 12 grocery products, ids 2..13.
    "product14": (
        "background",
        "undefined",
        "sprite",
        "kool-aid",
        "extra",
        "ocelo",
        "finish",
        "mtn_dew",
        "best_foods",
        "gatorade",
        "heinz",
        "ruffles",
        "pringles",
        "del_monte",
    ),
    # road-segmentation-adas-0001: four probability channels.
    "road4": ("background", "road", "curb", "lane mark"),
    # open-closed-eye-0001: softmax over [open, closed].
    "open_closed_eye": ("open", "closed"),
    # weld-porosity-detection-0001: logits over three states.
    "weld3": ("no weld", "normal weld", "porosity"),
    # common-sign-language-0002: 12 single-hand gestures.
    "sign_language12": (
        "digit 0",
        "digit 1",
        "digit 2",
        "digit 3",
        "digit 4",
        "digit 5",
        "thumb up",
        "thumb down",
        "sliding two fingers up",
        "sliding two fingers down",
        "sliding two fingers left",
        "sliding two fingers right",
    ),
    # person-attributes-recognition-crossroad-0234: independent sigmoids.
    "person_attributes7": (
        "male",
        "bag",
        "hat",
        "long sleeves",
        "long pants",
        "long hair",
        "coat/jacket",
    ),
    # vehicle-attributes-recognition-barrier-0042: two heads.
    "vehicle_type": ("car", "bus", "truck", "van"),
    "vehicle_color": ("white", "gray", "yellow", "red", "green", "blue", "black"),
    # face attribute heads shared by several OMZ models.
    "gender": ("female", "male"),
    "emotions5": ("neutral", "happy", "sad", "surprise", "anger"),
    "anti_spoof": ("real", "spoof"),
}


#: Class tables kept as data files rather than Python literals (ImageNet's 1000
#: names would dwarf this module), loaded on first use.
_FILE_TABLES = {"imagenet1000": "imagenet1000.txt"}


@cache
def _file_table(key: str) -> tuple[str, ...]:
    path = Path(__file__).resolve().parent.parent / "data" / _FILE_TABLES[key]
    lines = path.read_text(encoding="utf-8").splitlines()
    return tuple(ln.strip() for ln in lines if ln.strip())


def class_names(key: str | None, num_classes: int | None = None) -> dict[int, str]:
    """Resolve a manifest ``classes`` key to an ``{id: name}`` mapping.

    Falls back to ``class_<i>`` names when ``key`` is unknown. ``num_classes``,
    when given, sizes the fallback table. A table one shorter than the model's
    output (the classic ImageNet "background at index 0" variant) is shifted so
    the names still line up with the ids.
    """
    table: tuple[str, ...] | None = None
    if key and key in CLASS_TABLES:
        table = CLASS_TABLES[key]
    elif key and key in _FILE_TABLES:
        table = _file_table(key)
    if table is not None:
        offset = 1 if num_classes == len(table) + 1 else 0
        names = {i + offset: n for i, n in enumerate(table)}
        if offset:
            names[0] = "background"
        return names
    n = num_classes if num_classes is not None else 0
    return {i: f"class_{i}" for i in range(n)}
