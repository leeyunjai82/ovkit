"""Bring a Hugging Face model into ovkit as OpenVINO IR.

The Open Model Zoo is an archive; the models people actually want are on the
Hub. ``optimum-intel`` converts almost any of them to OpenVINO IR, so ovkit
does not need to mirror a model to serve it::

    ovkit pull google/vit-base-patch16-224
    Model("google/vit-base-patch16-224", "photo.jpg")   # cat 0.94

The split matters: **converting** needs the heavy half (torch, transformers,
optimum-intel — ``pip install "ovkit[hf]"``) and happens once; **running** the
converted IR afterwards needs nothing but ovkit, because what lands in the
cache is a plain OpenVINO model. A classroom machine can be handed the cache
and never see torch.

Alongside the IR, the pull writes what ovkit needs to answer in words rather
than indices: ``labels.txt`` from the model's own ``id2label``, and an
``ovkit.yaml`` carrying the resize and normalisation its image processor
declares — get those wrong and a perfectly good model returns confident
nonsense.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import yaml

from .core.constants import cache_root
from .core.errors import OVKitError
from .core.i18n import lang

#: HF task -> the optimum-intel class that exports it, and the ovkit task.
TASKS: dict[str, tuple[str, str]] = {
    "image-classification": ("OVModelForImageClassification", "classify"),
    "zero-shot-image-classification": ("OVModelForZeroShotImageClassification", "classify"),
    "feature-extraction": ("OVModelForFeatureExtraction", "classify"),
}
# Detection and segmentation are deliberately absent: optimum-intel exports
# neither, so ovkit serves those from the registry (RT-DETR, PSPNet) instead of
# advertising a conversion that cannot happen.

#: What `Model()` recognises as a Hub id rather than a registry name.
_HUB_ID = re.compile(r"^[\w.-]+/[\w.-]+$")


def is_hub_id(name: str) -> bool:
    """True for ``owner/model``; registry names never contain a slash."""
    return bool(_HUB_ID.match(str(name).strip()))


def local_dir(model_id: str) -> Path:
    """Where a pulled model lives: one directory per Hub id."""
    return cache_root() / "hf" / str(model_id).strip().replace("/", "__")


def pulled_models() -> list[str]:
    """Hub ids already converted into the cache, as you would type them."""
    root = cache_root() / "hf"
    if not root.is_dir():
        return []
    return sorted(
        d.name.replace("__", "/", 1)
        for d in root.iterdir()
        if d.is_dir() and (d / "openvino_model.xml").is_file()
    )


def ir_path(model_id: str) -> Path | None:
    """The converted ``.xml`` for ``model_id``, or ``None`` if not pulled yet."""
    xml = local_dir(model_id) / "openvino_model.xml"
    return xml if xml.is_file() else None


def not_pulled_message(model_id: str) -> str:
    """What to tell someone who used a Hub id before converting it."""
    if lang() == "ko":
        return (
            f"'{model_id}'는 아직 변환되지 않았어요.\n"
            f'  pip install "ovkit[hf]"  &&  ovkit pull {model_id}\n'
            f"한 번 변환하면 그 뒤로는 ovkit만으로 돌아갑니다."
        )
    return (
        f"'{model_id}' has not been converted yet.\n"
        f'  pip install "ovkit[hf]"  &&  ovkit pull {model_id}\n'
        f"After that it runs with nothing but ovkit."
    )


# -- pulling ----------------------------------------------------------------


def pull(model_id: str, task: str | None = None, precision: str = "fp16") -> Path:
    """Convert a Hub model to OpenVINO IR in the ovkit cache; return the .xml.

    ``task`` is read from the model's own config when omitted.
    """
    optimum = _require_optimum()
    resolved = task or detect_task(model_id)
    if resolved not in TASKS:
        raise OVKitError(
            f"ovkit does not serve the '{resolved}' task yet. "
            f"Supported: {', '.join(sorted(TASKS))}."
        )
    class_name, ovkit_task = TASKS[resolved]
    loader = getattr(optimum, class_name, None)
    if loader is None:  # pragma: no cover - optimum dropped the class
        raise OVKitError(f"optimum-intel has no {class_name}; is it up to date?")

    out = local_dir(model_id)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[ovkit] converting {model_id} ({resolved}) — this happens once...")
    model = loader.from_pretrained(model_id, export=True, compile=False)
    model.save_pretrained(out)

    _write_labels(out, getattr(model, "config", None))
    _write_sidecar(out, model_id=model_id, task=ovkit_task, precision=precision)
    xml = out / "openvino_model.xml"
    if not xml.is_file():  # pragma: no cover - optimum changed its filename
        found = next(out.glob("*.xml"), None)
        if found is None:
            raise OVKitError(f"conversion produced no IR in {out}")
        found.rename(xml)
        found.with_suffix(".bin").rename(xml.with_suffix(".bin"))
    print(f"[ovkit] ready: Model({model_id!r}, 'photo.jpg')")
    return xml


def detect_task(model_id: str) -> str:
    """The model's task, from its Hub config (``image-classification`` etc.)."""
    try:
        from transformers import AutoConfig
    except ImportError as exc:
        raise _missing_extra() from exc
    config = AutoConfig.from_pretrained(model_id)
    architectures = list(getattr(config, "architectures", None) or [])
    joined = " ".join(architectures).lower()
    if "zeroshot" in joined or "clip" in joined:
        return "zero-shot-image-classification"
    if "imageclassification" in joined:
        return "image-classification"
    if getattr(config, "id2label", None):
        return "image-classification"
    return "feature-extraction"


def _require_optimum() -> Any:
    try:
        import optimum.intel as optimum
    except ImportError as exc:
        raise _missing_extra() from exc
    return optimum


def _missing_extra() -> OVKitError:
    if lang() == "ko":
        return OVKitError(
            "허깅페이스 모델 변환에는 추가 설치가 필요해요:\n"
            '  pip install "ovkit[hf]"\n'
            "변환은 한 번만 하면 되고, 그 뒤 실행에는 필요 없습니다."
        )
    return OVKitError(
        "Converting a Hugging Face model needs the extra:\n"
        '  pip install "ovkit[hf]"\n'
        "It is needed for the conversion only, never to run the result."
    )


def _write_labels(out: Path, config: Any = None) -> None:
    """Write labels.txt from the model's id2label, if it has one."""
    table = dict(getattr(config, "id2label", None) or {})
    if not table:
        return
    names = [str(table.get(i, table.get(str(i), f"class_{i}"))) for i in range(len(table))]
    (out / "labels.txt").write_text("\n".join(names) + "\n", encoding="utf-8")


def _write_sidecar(out: Path, *, model_id: str, task: str, precision: str) -> None:
    """Record the preprocessing the model's own image processor declares.

    A ViT wants (x/255 - 0.5) / 0.5 and a ConvNeXt wants ImageNet statistics;
    guessing here is the difference between an answer and confident nonsense.
    """
    processor = out / "preprocessor_config.json"
    pre: dict[str, Any] = {"rgb": True, "scale": 255.0}
    if processor.is_file():
        try:
            cfg = json.loads(processor.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cfg = {}
        if not cfg.get("do_rescale", True):
            pre["scale"] = 1.0
        if cfg.get("do_normalize", True):
            if cfg.get("image_mean"):
                pre["mean"] = [float(v) for v in cfg["image_mean"]]
            if cfg.get("image_std"):
                pre["std"] = [float(v) for v in cfg["image_std"]]
    (out / "ovkit.yaml").write_text(
        yaml.safe_dump(
            {
                "model_id": model_id,
                "task": task,
                "precision": precision,
                "preprocess": pre,
                "postprocess": {"softmax": True},
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )


def sidecar(model_id: str) -> dict[str, Any]:
    """The ``ovkit.yaml`` written at pull time, or ``{}``."""
    path = local_dir(model_id) / "ovkit.yaml"
    if not path.is_file():
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, ValueError):  # pragma: no cover - corrupt sidecar
        return {}
