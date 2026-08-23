"""genai registry wiring (no openvino-genai / network needed)."""

from __future__ import annotations

import pytest

from ovkit import Model, OVKitError
from ovkit.core.registry import resolve


def test_genai_entries_registered():
    entry = resolve("whisper_base")
    assert entry is not None
    assert entry.src == "genai"
    assert entry.extra.get("pipeline") == "whisper"
    # The exact SPDX id follows upstream (HF reports apache-2.0 for whisper-base);
    # what ovkit guarantees is that it stays permissive.
    from ovkit.core.constants import is_permissive

    assert is_permissive(entry.license)


def test_model_redirects_genai_to_pipeline():
    # Model() must not try to load a genai model as IR; it points to ovkit.genai.
    with pytest.raises(OVKitError):
        Model("tinyllama_chat")
