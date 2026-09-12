"""The mirror audit decides what gets deleted, so its keep-rules are tested.

A bug here is not a wrong answer on screen: it is a model missing from the one
repository ovkit downloads from.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parent.parent / "scripts" / "audit_mirror.py"


@pytest.fixture(scope="module")
def audit():
    spec = importlib.util.spec_from_file_location("audit_mirror", _PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["audit_mirror"] = module
    spec.loader.exec_module(module)
    return module


def test_manifests_reference_the_mirror(audit):
    """The real manifests must resolve to real mirror paths."""
    files, prefixes = audit.referenced()
    assert "detect/rtdetr_r18/model.xml" in files
    assert "detect/rtdetr_r18/model.bin" in files, "IR weights travel with the .xml"
    assert "depth/depth_anything_v2_small/model.xml" in files
    assert "background/u2net/u2net.onnx" in files
    assert "genai/whisper_base/" in prefixes, "genai pipelines download whole folders"


def test_companions_ride_along_with_a_served_model(audit):
    """A kept model keeps its class names and its provenance."""
    files, prefixes = audit.referenced()
    for name in ("labels.txt", "README.md", "LICENSE", "LICENSE.md"):
        assert audit._kept(f"detect/rtdetr_r18/{name}", files, prefixes), name


def test_companions_go_with_a_dropped_model(audit):
    """They are provenance for the model, not furniture that outlives it."""
    files, prefixes = audit.referenced()
    assert not audit._kept("detect/model_we_dropped/README.md", files, prefixes)
    assert not audit._kept("detect/model_we_dropped/LICENSE", files, prefixes)


def test_a_genai_subtree_is_kept_whole(audit):
    """A genai pipeline is a directory of files no manifest names one by one."""
    files, prefixes = audit.referenced()
    assert audit._kept("genai/whisper_base/openvino_encoder_model.bin", files, prefixes)
    assert audit._kept("genai/whisper_base/tokenizer_config.json", files, prefixes)


def test_provenance_and_furniture_survive(audit):
    files, prefixes = audit.referenced()
    assert audit._kept("README.md", files, prefixes), "the mirror's own front page"
    assert audit._kept(".gitattributes", files, prefixes)


def test_an_unreferenced_file_is_an_orphan(audit):
    files, prefixes = audit.referenced()
    assert not audit._kept("detect/some_model_we_dropped/model.xml", files, prefixes)
    assert not audit._kept("depth/depth_anything_v2_large/model.bin", files, prefixes)


def test_empty_manifests_never_prune_the_whole_mirror(audit, monkeypatch, capsys):
    """No manifests read (wrong cwd, bad checkout) must not mean 'delete everything'."""
    monkeypatch.setattr(audit, "referenced", lambda: (set(), set()))
    status, orphans, _ = audit.audit()
    assert status == 2 and orphans == []
    assert "refusing" in capsys.readouterr().err


def test_report_separates_dead_models_from_leftovers(audit, capsys):
    """Two piles, two decisions: a model that goes, versus a file beside one that stays."""
    files = {"detect/live/model.xml", "detect/live/model.bin"}
    orphans = [
        ("detect/live/original-name.xml", 40_000),  # leftover beside a served model
        ("detect/gone/model.xml", 1_000),
        ("detect/gone/model.bin", 9_000_000),
    ]
    audit._report(orphans, files)
    out = capsys.readouterr().out
    assert "still serves" in out and "live/original-name.xml" in out
    dead = out.split("no manifest mentions")[1]
    assert "gone(" in dead, "the retired model is named, with its size"
    assert "live" not in dead, "the served model is not in the pile that gets retired"


def test_the_safe_half_can_be_pruned_on_its_own(audit):
    """Deleting a duplicate beside a live model and retiring a model are not
    the same decision, so they are not the same command."""
    files = {"detect/live/model.xml"}
    orphans = [("detect/live/old-name.xml", 40), ("detect/gone/model.xml", 9)]
    piles = audit.split(orphans, files)
    assert piles["leftovers"] == [("detect/live/old-name.xml", 40)]
    assert piles["models"] == [("detect/gone/model.xml", 9)]
