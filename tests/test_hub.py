"""Pulling a Hugging Face model: convert once, then run with plain ovkit."""

from __future__ import annotations

import json

import pytest
import yaml

from ovkit import Model, hub
from ovkit.core.errors import ModelNotFoundError, OVKitError


@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setenv("OVKIT_HOME", str(tmp_path))
    return tmp_path


# -- recognising a Hub id ---------------------------------------------------


@pytest.mark.parametrize(
    "name,expected",
    [
        ("google/vit-base-patch16-224", True),
        ("openai/clip-vit-base-patch32", True),
        ("detect", False),  # a registry name never has a slash
        ("face_analyze", False),
        ("a/b/c", False),  # not a Hub id shape
        ("/leading", False),
        ("model.xml", False),
    ],
)
def test_hub_ids_are_told_apart_from_registry_names(name, expected):
    assert hub.is_hub_id(name) is expected


def test_a_hub_id_never_collides_with_a_registry_name():
    from ovkit.core.registry import list_models

    assert not [n for n in list_models(tier=None) if hub.is_hub_id(n)]


# -- using one before converting it -----------------------------------------


def test_an_unpulled_model_says_how_to_convert_it(cache, monkeypatch):
    monkeypatch.setenv("OVKIT_LANG", "en")
    with pytest.raises(ModelNotFoundError, match=r"ovkit pull google/vit"):
        Model("google/vit-base-patch16-224")


def test_the_same_message_in_korean(cache, monkeypatch):
    monkeypatch.setenv("OVKIT_LANG", "ko")
    with pytest.raises(ModelNotFoundError, match="변환되지 않았"):
        Model("google/vit-base-patch16-224")


def test_converting_without_the_extra_names_it(monkeypatch):
    monkeypatch.setenv("OVKIT_LANG", "en")
    monkeypatch.setattr(
        hub, "_require_optimum", lambda: (_ for _ in ()).throw(hub._missing_extra())
    )
    with pytest.raises(OVKitError, match=r"ovkit\[hf\]"):
        hub.pull("google/vit-base-patch16-224", task="image-classification")


def test_an_unsupported_task_lists_the_supported_ones(monkeypatch):
    monkeypatch.setattr(hub, "_require_optimum", lambda: object())
    with pytest.raises(OVKitError, match="image-classification"):
        hub.pull("some/model", task="text-generation")


# -- what a pull leaves behind ----------------------------------------------


def _fake_pulled(cache, model_id: str, *, labels=("cat", "dog"), processor=None) -> None:
    """Stand in for a completed conversion."""
    out = hub.local_dir(model_id)
    out.mkdir(parents=True, exist_ok=True)
    (out / "openvino_model.xml").write_text("<net/>")
    (out / "openvino_model.bin").write_bytes(b"\0")
    if labels:
        (out / "labels.txt").write_text("\n".join(labels) + "\n")
    if processor is not None:
        (out / "preprocessor_config.json").write_text(json.dumps(processor))


def test_labels_come_from_the_models_own_id2label(cache):
    out = hub.local_dir("x/y")
    out.mkdir(parents=True)

    class _Config:
        id2label = {0: "tench", 1: "goldfish"}

    hub._write_labels(out, _Config())
    assert (out / "labels.txt").read_text().splitlines() == ["tench", "goldfish"]


def test_a_model_without_labels_writes_no_sidecar(cache):
    out = hub.local_dir("x/y")
    out.mkdir(parents=True)
    hub._write_labels(out, object())
    assert not (out / "labels.txt").exists()


def test_the_sidecar_records_the_processors_normalisation(cache):
    """ViT wants (x/255 - 0.5)/0.5; guessing here returns confident nonsense."""
    out = hub.local_dir("x/y")
    out.mkdir(parents=True)
    (out / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "do_rescale": True,
                "do_normalize": True,
                "image_mean": [0.5, 0.5, 0.5],
                "image_std": [0.5, 0.5, 0.5],
            }
        )
    )
    hub._write_sidecar(out, model_id="x/y", task="classify", precision="fp16")
    meta = yaml.safe_load((out / "ovkit.yaml").read_text())
    assert meta["preprocess"] == {
        "rgb": True,
        "scale": 255.0,
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
    }
    assert meta["task"] == "classify"


def test_a_processor_that_does_not_rescale_keeps_raw_pixels(cache):
    out = hub.local_dir("x/y")
    out.mkdir(parents=True)
    (out / "preprocessor_config.json").write_text(
        json.dumps({"do_rescale": False, "do_normalize": False})
    )
    hub._write_sidecar(out, model_id="x/y", task="classify", precision="fp16")
    meta = yaml.safe_load((out / "ovkit.yaml").read_text())
    assert meta["preprocess"]["scale"] == 1.0
    assert "mean" not in meta["preprocess"]


# -- running a pulled model -------------------------------------------------


def test_a_pulled_model_is_listed_and_found(cache):
    assert hub.pulled_models() == []
    _fake_pulled(cache, "google/vit-base-patch16-224")
    assert hub.pulled_models() == ["google/vit-base-patch16-224"]
    assert hub.ir_path("google/vit-base-patch16-224").name == "openvino_model.xml"


def test_model_resolves_a_pulled_id_to_its_cached_ir(cache):
    _fake_pulled(cache, "google/vit-base-patch16-224")
    hub._write_sidecar(
        hub.local_dir("google/vit-base-patch16-224"),
        model_id="google/vit-base-patch16-224",
        task="classify",
        precision="fp16",
    )
    model = Model.network("google/vit-base-patch16-224")
    assert model.ir_path == hub.ir_path("google/vit-base-patch16-224")
    assert model._pre["scale"] == 255.0  # from the sidecar, not a guess


def test_running_a_pulled_model_needs_no_optimum(cache, monkeypatch):
    """The cache holds plain IR — importing optimum at run time would be a bug."""
    import builtins

    _fake_pulled(cache, "google/vit-base-patch16-224")
    real_import = builtins.__import__

    def no_optimum(name, *args, **kwargs):
        if name.startswith("optimum") or name.startswith("transformers"):
            raise AssertionError(f"running a pulled model imported {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_optimum)
    Model.network("google/vit-base-patch16-224")


def test_every_supported_task_names_a_real_optimum_class():
    optimum = pytest.importorskip("optimum.intel")
    for task, (class_name, _ovkit_task) in hub.TASKS.items():
        assert hasattr(optimum, class_name), f"{task} -> {class_name} is gone"
