"""openvino-genai models, callable in one line (optional ``ovkit[genai]`` extra).

Generation itself is openvino-genai's; ovkit resolves the model name, downloads
it from the mirror, and converts your input into the tensor the pipeline wants::

    from ovkit.genai import generate, describe, transcribe

    generate("Explain OpenVINO in one sentence.")   # LLM   -> str
    describe("desk.jpg", "How many monitors?")      # VLM   -> str
    transcribe("meeting.wav")                       # Whisper -> str

:func:`pipeline` still returns the raw ``openvino_genai`` pipeline when you want
streaming, chat state, or generation configs.
"""

from __future__ import annotations

from .pipelines import (
    describe,
    generate,
    llm_pipeline,
    pipeline,
    text2image_pipeline,
    text2speech_pipeline,
    transcribe,
    vlm_pipeline,
    whisper_pipeline,
)

__all__ = [
    "pipeline",
    "generate",
    "describe",
    "transcribe",
    "llm_pipeline",
    "text2image_pipeline",
    "whisper_pipeline",
    "vlm_pipeline",
    "text2speech_pipeline",
]
