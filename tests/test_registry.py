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


def test_spdx_from_license_url_identifies_permissive_urls():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from build_mirror import _spdx_from_license_url

    assert _spdx_from_license_url("https://www.apache.org/licenses/LICENSE-2.0") == "apache-2.0"
    assert _spdx_from_license_url("https://opensource.org/licenses/MIT") == "mit"
    assert _spdx_from_license_url("https://opensource.org/licenses/BSD-3-Clause") == "bsd-3-clause"
    # Non-permissive or unknown must be None so the mirror skips it.
    assert _spdx_from_license_url("https://creativecommons.org/licenses/by-nc/4.0/") is None
    assert _spdx_from_license_url(None) is None
