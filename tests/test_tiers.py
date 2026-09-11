"""Model tiers: a curated front page, with the zoo still reachable by name."""

from __future__ import annotations

import pytest

from ovkit import Model
from ovkit.core import registry
from ovkit.core.registry import list_models, resolve, tier_of
from ovkit.pipelines import PIPELINES

#: Names the audit found unreachable from any capability, alias or the GUI.
ARCHIVED = [
    "weld_porosity_detection_0001",
    "product_detection_0001",
    "yolo_v3_onnx",
    "machine_translation_nar_en_de_0002",
    "aclnet",
    "tinyllama_chat",
    "whisper_large_v3_turbo",
]


@pytest.mark.parametrize("name", ARCHIVED)
def test_archived_models_are_off_the_front_page(name):
    assert tier_of(name) == "zoo"
    assert name not in list_models()


@pytest.mark.parametrize("name", ARCHIVED)
def test_archived_models_still_load_by_name(name):
    """Tiering is about what ovkit puts in front of you, not what it can run."""
    assert name in list_models(tier=None)
    assert resolve(name) is not None


def test_the_curated_set_is_small_enough_to_read():
    core = list_models()
    assert 30 <= len(core) <= 70, f"{len(core)} names on the front page"
    assert len(core) < len(list_models(tier=None))


def test_an_alias_is_only_as_visible_as_its_target():
    """Hiding a model but leaving its friendly name would move the dead end."""
    assert tier_of("sound_classification") == "zoo"  # -> aclnet
    assert tier_of("qa") == "zoo"  # -> bert squad
    assert tier_of("detect") == "core"  # -> rtdetr_r50


def test_an_alias_cycle_does_not_hang(monkeypatch):
    raw = dict(registry._load_raw())
    raw.update({"loop_a": {"alias": "loop_b"}, "loop_b": {"alias": "loop_a"}})
    monkeypatch.setattr(registry, "_load_raw", lambda: raw)
    assert tier_of("loop_a") == "core"


def _models_each_capability_uses() -> dict[str, set[str]]:
    """Registry names every pipeline reaches, from defaults and self.model()."""
    import inspect
    import re
    from pathlib import Path

    sources = {
        path.stem: path.read_text()
        for path in (Path(__import__("ovkit").__file__).parent / "pipelines").glob("*.py")
    }
    used: dict[str, set[str]] = {}
    for name, cls in PIPELINES.items():
        names = {
            p.default
            for p in inspect.signature(cls.__init__).parameters.values()
            if isinstance(p.default, str) and resolve(p.default) is not None
        }
        module = cls.__module__.rsplit(".", 1)[-1]
        names |= {
            literal
            for literal in re.findall(r'self\.model\(\s*"([^"]+)"\s*\)', sources.get(module, ""))
            if resolve(literal) is not None
        }
        used[name] = names
    return used


def test_no_capability_is_built_on_an_archived_model():
    """The front page must not advertise something assembled from the attic."""
    offenders = {
        cap: sorted(m for m in models if tier_of(m) == "zoo")
        for cap, models in _models_each_capability_uses().items()
    }
    offenders = {cap: models for cap, models in offenders.items() if models}
    assert not offenders, f"capabilities built on archived models: {offenders}"


def test_every_capability_reaches_at_least_one_real_model():
    """Catches a capability whose sub-model names were renamed out from under it."""
    used = _models_each_capability_uses()
    # face_match and teach take their embedder by name; anomaly takes yours.
    standalone = {"anomaly"}
    for cap, models in used.items():
        if cap in standalone:
            continue
        assert models, f"{cap} reaches no registered model"


def test_default_genai_names_survive_the_cut():
    for alias, target in (("llm", "qwen25_1_5b_instruct"), ("stt", "whisper_base")):
        assert tier_of(alias) == "core"
        assert resolve(alias).name == target


def test_zoo_models_are_still_callable_through_Model():
    """Model(name) must not care about tiers — only listings do."""
    from ovkit.core.errors import ModelNotFoundError

    try:
        Model("weld_porosity_detection_0001")
    except ModelNotFoundError:
        pytest.fail("a zoo model should still be resolvable by name")
    except Exception:
        pass  # download/compile is fine to fail here; resolution is the point
