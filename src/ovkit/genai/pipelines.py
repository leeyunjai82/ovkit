"""Thin wrappers over openvino-genai pipelines (optional ``ovkit[genai]``).

Each factory forwards to the corresponding ``openvino_genai`` pipeline, only
unifying the default device. The generation pipelines themselves are **not**
reimplemented (spec §11).
"""

from __future__ import annotations

from typing import Any

_DEFAULT_DEVICE = "AUTO"


def _require_genai() -> Any:
    try:
        import openvino_genai as ovg
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "openvino-genai is required. Install it with: pip install 'ovkit[genai]'."
        ) from exc
    return ovg


def llm_pipeline(model_path: str, device: str = _DEFAULT_DEVICE, **kwargs: Any) -> Any:
    """Return an ``openvino_genai.LLMPipeline``."""
    return _require_genai().LLMPipeline(model_path, device, **kwargs)


def text2image_pipeline(model_path: str, device: str = _DEFAULT_DEVICE, **kwargs: Any) -> Any:
    """Return an ``openvino_genai.Text2ImagePipeline``."""
    return _require_genai().Text2ImagePipeline(model_path, device, **kwargs)


def whisper_pipeline(model_path: str, device: str = _DEFAULT_DEVICE, **kwargs: Any) -> Any:
    """Return an ``openvino_genai.WhisperPipeline``."""
    return _require_genai().WhisperPipeline(model_path, device, **kwargs)


def vlm_pipeline(model_path: str, device: str = _DEFAULT_DEVICE, **kwargs: Any) -> Any:
    """Return an ``openvino_genai.VLMPipeline``."""
    return _require_genai().VLMPipeline(model_path, device, **kwargs)


def text2speech_pipeline(model_path: str, device: str = _DEFAULT_DEVICE, **kwargs: Any) -> Any:
    """Return an ``openvino_genai.Text2SpeechPipeline``."""
    return _require_genai().Text2SpeechPipeline(model_path, device, **kwargs)
