"""Thin re-exports of openvino-genai pipelines (optional ``ovkit[genai]`` extra).

We do not reimplement generation; we only unify the default device. Import the
helpers from :mod:`ovkit.genai.pipelines`.
"""

from __future__ import annotations

from .pipelines import (
    llm_pipeline,
    text2image_pipeline,
    text2speech_pipeline,
    vlm_pipeline,
    whisper_pipeline,
)

__all__ = [
    "llm_pipeline",
    "text2image_pipeline",
    "whisper_pipeline",
    "vlm_pipeline",
    "text2speech_pipeline",
]
