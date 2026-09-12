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


# -- the RT-DETR ladder -----------------------------------------------------


@pytest.mark.parametrize("name", ["rtdetr_r18", "rtdetr_r34", "rtdetr_r50"])
def test_the_detection_ladder_is_registered_and_labelled(name):
    entry = resolve(name)
    assert entry is not None, f"{name} missing"
    assert entry.repo == "leeyunjai/ovkit-models", "one mirror for everything"
    assert entry.filename == f"detect/{name}/model.xml"
    # COCO names, or every box answers "class_21"
    assert entry.postprocess.get("classes") == "coco80"
    assert entry.postprocess.get("format") == "detr"
    assert tier_of(name) == "core"


def test_detect_defaults_to_the_one_that_keeps_up_with_a_webcam():
    """r50 benchmarks at ~2 FPS on CPU; a classroom default cannot be that."""
    assert resolve("detect").name == "rtdetr_r18"


# -- one mirror -------------------------------------------------------------


def _primary_source(entry) -> str:
    return entry.repo or entry.url or ""


def test_every_model_is_served_from_the_one_mirror():
    """ovkit downloads from one repository, so a school keeps one copy alive."""
    from ovkit.core import registry

    raw = registry._load_raw()
    strays = []
    for name in sorted(raw):
        if "alias" in raw[name]:
            continue
        entry = resolve(name)
        if entry is None:
            continue
        if "leeyunjai/ovkit-models" not in _primary_source(entry):
            strays.append((name, _primary_source(entry)))
    assert not strays, f"models served from somewhere else: {strays}"


def test_nothing_is_served_straight_out_of_the_agpl_repo():
    """edge-lab is tagged agpl-3.0; its Apache models are copied, not linked."""
    from ovkit.core import registry

    for name in registry.list_models(tier=None):
        entry = resolve(name)
        if entry is None:
            continue
        both = f"{_primary_source(entry)} {(entry.fallback or {}).get('repo', '')}"
        assert "edge-lab" not in both, f"{name} reads from edge-lab"


def test_models_that_originate_elsewhere_keep_their_home_as_fallback():
    for name in ("rtdetr_r18", "rtdetr_r34", "rtdetr_r50"):
        entry = resolve(name)
        assert (entry.fallback or {}).get("repo") == "leeyunjai/rtdetr"


def test_the_sync_script_covers_every_model_it_should():
    """A manifest pointing at a mirror path nobody copies there is a dead link."""
    import sys
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__import__("ovkit").__file__).parents[2] / "scripts"))
    import sync_mirror

    copied = {dest for dest, _what, _lic in sync_mirror.FILES.values()}
    for name in ("rtdetr_r18", "rtdetr_r34", "rtdetr_r50", "depth_anything_v2_small", "u2net"):
        entry = resolve(name)
        assert entry.filename in copied, f"{name} -> {entry.filename} is never synced"


def test_the_readme_model_count_is_the_real_one():
    """The front page says how many models ovkit serves; it has to still be true.

    That number was written once and went stale — it read 56 while the registry
    held 67. A count nobody checks is a claim nobody can trust, so the check
    lives here instead of in someone's memory.
    """
    import re
    from pathlib import Path

    from ovkit.core import registry

    raw = registry._load_raw()
    models = [n for n in registry.list_models(tier=None) if "alias" not in raw[n]]

    readme = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
    claimed = re.search(r"over (\d+) ready models", readme)
    assert claimed, "the README no longer states a model count"
    assert int(claimed.group(1)) == len(
        models
    ), f"README says {claimed.group(1)} models, the registry holds {len(models)}"
