"""Download cache: offline mode, atomic url fetch, integrity checks."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from ovkit.core import download
from ovkit.core.errors import DownloadError, OfflineError
from ovkit.core.registry import ModelEntry


def test_cache_dir_honors_ovkit_home(tmp_path, monkeypatch):
    monkeypatch.setenv("OVKIT_HOME", str(tmp_path))
    d = download.model_cache_dir("foo")
    assert str(d).startswith(str(tmp_path))
    assert d.is_dir()


def test_atomic_url_download_and_verify(tmp_path):
    payload = b"hello-ovkit"
    src = tmp_path / "src.bin"
    src.write_bytes(payload)
    dest = tmp_path / "out" / "dest.bin"

    download._atomic_url_download(src.as_uri(), dest)
    assert dest.read_bytes() == payload
    # no leftover temp files
    assert not list(dest.parent.glob("*.part"))

    digest = hashlib.sha256(payload).hexdigest()
    download._verify(dest, digest)  # matches -> no error
    with pytest.raises(DownloadError):
        download._verify(dest, "0" * 64)  # mismatch -> raises and removes
    assert not dest.exists()


def test_offline_without_cache_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("OVKIT_HOME", str(tmp_path))
    monkeypatch.setenv("OVKIT_OFFLINE", "1")
    entry = ModelEntry(name="ghost", src="hf", repo="x/y", filename="m.onnx")
    with pytest.raises(OfflineError):
        download.fetch(entry)


def test_offline_uses_cached_source(tmp_path, monkeypatch):
    monkeypatch.setenv("OVKIT_HOME", str(tmp_path))
    entry = ModelEntry(name="cached", src="hf", repo="x/y", filename="m.onnx")
    src_dir = download.downloads_dir("cached")
    (src_dir / "m.onnx").write_bytes(b"fake-onnx")

    monkeypatch.setenv("OVKIT_OFFLINE", "1")
    found = download.fetch(entry)
    assert Path(found).name == "m.onnx"
