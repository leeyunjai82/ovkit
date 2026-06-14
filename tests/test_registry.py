"""Registry + license-policy tests."""

from __future__ import annotations

import pytest

from ovkit.core import registry
from ovkit.core.errors import LicenseError


def test_bundled_models_listed():
    names = registry.list_models()
    assert "rtdetr_r50" in names


def test_resolve_known_model():
    entry = registry.resolve("rtdetr_r50")
    assert entry is not None
    assert entry.task == "detect"
    assert entry.license == "apache-2.0"
    assert entry.src == "hf"


def test_resolve_unknown_returns_none():
    assert registry.resolve("does-not-exist-xyz") is None


def test_non_permissive_license_rejected(tmp_path, monkeypatch):
    manifest = tmp_path / "bad.yaml"
    manifest.write_text(
        "evil_model:\n  src: hf\n  repo: someone/evil\n  task: detect\n" "  license: agpl-3.0\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OVKIT_MANIFESTS", str(tmp_path))
    registry.reload()
    try:
        assert "evil_model" in registry.list_models()
        with pytest.raises(LicenseError):
            registry.resolve("evil_model")
    finally:
        monkeypatch.delenv("OVKIT_MANIFESTS", raising=False)
        registry.reload()


def test_alias_resolves_to_target(tmp_path, monkeypatch):
    m = tmp_path / "alias_test.yaml"
    m.write_text(
        "real_det:\n  src: hf\n  repo: x/y\n  filename: a.xml\n  task: detect\n"
        "  license: apache-2.0\n"
        "detect_alias:\n  alias: real_det\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("OVKIT_MANIFESTS", str(m))
    registry.reload()
    entry = registry.resolve("detect_alias")
    assert entry is not None
    assert entry.name == "real_det"
    assert entry.task == "detect"
    registry.reload()
