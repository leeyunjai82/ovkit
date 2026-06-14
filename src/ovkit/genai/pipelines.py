"""openvino-genai pipelines, made usable: resolve a name, download, build.

``pipeline(name)`` resolves a registered genai model (``manifests/genai.yaml``),
downloads the OpenVINO-converted model from Hugging Face, and returns the ready
``openvino_genai`` pipeline. A local directory path also works.

    from ovkit.genai import pipeline
    llm = pipeline("tinyllama_chat")
    print(llm.generate("Hello", max_new_tokens=50))

    stt = pipeline("whisper_base")
    print(stt.generate(audio))           # audio: float32 16 kHz mono ndarray

Requires ``pip install -e ".[genai]"`` (openvino-genai + optimum-intel).

The thin per-type factories (``llm_pipeline`` etc.) remain for building a
pipeline directly from a model directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.errors import OVKitError
from ..core.registry import resolve

_DEFAULT_DEVICE = "AUTO"


def _require_genai() -> Any:
    try:
        import openvino_genai as ovg
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "openvino-genai is required. Install it with: pip install 'ovkit[genai]'."
        ) from exc
    return ovg


def _builder(ptype: str):
    ovg = _require_genai()
    builders = {
        "llm": ovg.LLMPipeline,
        "whisper": ovg.WhisperPipeline,
        "text2image": ovg.Text2ImagePipeline,
        "text2speech": ovg.Text2SpeechPipeline,
        "vlm": ovg.VLMPipeline,
    }
    if ptype not in builders:
        raise OVKitError(f"Unknown genai pipeline type {ptype!r}. Known: {sorted(builders)}.")
    return builders[ptype]


def _download(repo: str) -> str:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - dependency missing
        raise OVKitError("huggingface_hub is required to download genai models.") from exc
    return snapshot_download(repo_id=repo)


def pipeline(name: str, device: str = _DEFAULT_DEVICE, **kwargs: Any) -> Any:
    """Build an openvino-genai pipeline from a registered name or a local path.

    Registered genai names (see ``manifests/genai.yaml``) are downloaded from
    Hugging Face; a local directory is used as-is (its pipeline type is then
    required via ``pipeline_type=...``).
    """
    entry = resolve(name)
    if entry is not None and entry.src == "genai":
        ptype = entry.extra.get("pipeline") or kwargs.pop("pipeline_type", None)
        if not ptype:
            raise OVKitError(f"genai model '{name}' is missing a 'pipeline' type.")
        model_dir = _download(entry.repo) if entry.repo else name
    else:
        model_dir = str(name)
        ptype = kwargs.pop("pipeline_type", None)
        if not ptype:
            raise OVKitError(
                f"'{name}' is not a registered genai model. For a local path, pass "
                f"pipeline_type='llm'|'whisper'|'text2image'|'text2speech'|'vlm'."
            )
        if not Path(model_dir).exists():
            raise OVKitError(f"genai model path not found: {model_dir}")
    return _builder(ptype)(model_dir, device, **kwargs)


# --- thin per-type factories (build directly from a model directory) -------


def llm_pipeline(model_path: str, device: str = _DEFAULT_DEVICE, **kwargs: Any) -> Any:
    """Return an ``openvino_genai.LLMPipeline``."""
    return _require_genai().LLMPipeline(model_path, device, **kwargs)


def text2image_pipeline(model_path: str, device: str = _DEFAULT_DEVICE, **kwargs: Any) -> Any:
    """Return an ``openvino_genai.Text2ImagePipeline``."""
    return _require_genai().Text2ImagePipeline(model_path, device, **kwargs)


def whisper_pipeline(model_path: str, device: str = _DEFAULT_DEVICE, **kwargs: Any) -> Any:
    """Return an ``openvino_genai.WhisperPipeline`` (speech-to-text)."""
    return _require_genai().WhisperPipeline(model_path, device, **kwargs)


def vlm_pipeline(model_path: str, device: str = _DEFAULT_DEVICE, **kwargs: Any) -> Any:
    """Return an ``openvino_genai.VLMPipeline``."""
    return _require_genai().VLMPipeline(model_path, device, **kwargs)


def text2speech_pipeline(model_path: str, device: str = _DEFAULT_DEVICE, **kwargs: Any) -> Any:
    """Return an ``openvino_genai.Text2SpeechPipeline`` (text-to-speech)."""
    return _require_genai().Text2SpeechPipeline(model_path, device, **kwargs)
