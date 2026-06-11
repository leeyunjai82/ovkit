"""Artifact download + cache with atomic writes, integrity checks, offline mode.

This module turns a :class:`~ovkit.core.registry.ModelEntry` into a concrete
local file (an ``.onnx`` or ``.xml``) ready for conversion/loading. It does not
know about OpenVINO; it only fetches bytes and verifies them.

Robustness guarantees (spec §6.5):

1. Atomic save  — download to a temp file, ``rename`` into place on success.
2. Integrity    — verify ``sha256`` from the manifest when present.
3. Offline      — ``OVKIT_OFFLINE=1`` blocks the network; cache-only.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from .constants import cache_root, is_offline
from .errors import DownloadError, GatedModelError, MirrorMissingError, OfflineError
from .registry import ModelEntry

#: HF mirror that hosts OMZ-derived (Apache-2.0) IR for ovkit.
OVKIT_MIRROR = "leeyunjai/ovkit-models"


def model_cache_dir(name: str) -> Path:
    """Return (and create) the cache directory for a model ``name``."""
    d = cache_root() / "models" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def downloads_dir(name: str) -> Path:
    """Return (and create) the directory holding raw downloaded sources."""
    d = model_cache_dir(name) / "src"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify(path: Path, expected: str | None) -> None:
    if not expected:
        return
    actual = _sha256(path)
    if actual.lower() != expected.lower():
        path.unlink(missing_ok=True)
        raise DownloadError(
            f"Checksum mismatch for {path.name}: expected {expected}, got {actual}. "
            f"The file has been removed; re-run to download again."
        )


def _atomic_url_download(url: str, dest: Path) -> None:
    """Download ``url`` to ``dest`` atomically (temp file + rename)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(dest.parent), suffix=".part")
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with urllib.request.urlopen(url) as resp, tmp.open("wb") as out:  # noqa: S310
            shutil.copyfileobj(resp, out)
        tmp.replace(dest)
    except urllib.error.HTTPError as exc:  # pragma: no cover - network
        tmp.unlink(missing_ok=True)
        raise DownloadError(f"HTTP {exc.code} while downloading {url}") from exc
    except urllib.error.URLError as exc:  # pragma: no cover - network
        tmp.unlink(missing_ok=True)
        raise DownloadError(f"Network error while downloading {url}: {exc.reason}") from exc
    finally:
        tmp.unlink(missing_ok=True)


def _fetch_hf(entry: ModelEntry, dest_dir: Path) -> Path:
    """Download a single file (or snapshot) from Hugging Face Hub."""
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
        from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
    except ImportError as exc:  # pragma: no cover - dependency missing
        raise DownloadError(
            "huggingface_hub is required to download '" + entry.name + "'. "
            "Install ovkit's core dependencies."
        ) from exc

    if not entry.repo:
        raise DownloadError(f"Model '{entry.name}' is missing a 'repo' field.")

    try:
        if entry.filename:
            path = hf_hub_download(
                repo_id=entry.repo,
                filename=entry.filename,
                subfolder=entry.subfolder,
                local_dir=str(dest_dir),
            )
            # OpenVINO IR is two files: pull the sibling .bin alongside the .xml
            # so the weights are present (a single-file fetch would miss them).
            if entry.filename.endswith(".xml"):
                bin_name = entry.filename[: -len(".xml")] + ".bin"
                try:
                    hf_hub_download(
                        repo_id=entry.repo,
                        filename=bin_name,
                        subfolder=entry.subfolder,
                        local_dir=str(dest_dir),
                    )
                except Exception:
                    pass  # weight-embedded IR or no companion .bin: tolerate
            return Path(path)
        snap = snapshot_download(repo_id=entry.repo, local_dir=str(dest_dir))
        return Path(snap)
    except GatedRepoError as exc:
        raise GatedModelError(
            f"Hugging Face repo '{entry.repo}' is gated. Request access on the model "
            f"page and authenticate with `huggingface-cli login` (or set the HF_TOKEN "
            f"environment variable), then retry."
        ) from exc
    except RepositoryNotFoundError as exc:
        if entry.repo == OVKIT_MIRROR:
            raise MirrorMissingError(
                f"'{entry.filename or entry.name}' is not yet available on the ovkit "
                f"mirror ({OVKIT_MIRROR}). This model still needs to be uploaded to "
                f"the mirror before it can be used."
            ) from exc
        raise DownloadError(
            f"Hugging Face repo '{entry.repo}' not found for model '{entry.name}'."
        ) from exc
    except Exception as exc:  # pragma: no cover - network/hub variety
        raise DownloadError(f"Failed to download '{entry.name}' from Hugging Face: {exc}") from exc


def _fetch_url(entry: ModelEntry, dest_dir: Path) -> Path:
    if not entry.url:
        raise DownloadError(f"Model '{entry.name}' is missing a 'url' field.")
    fname = entry.filename or entry.url.rsplit("/", 1)[-1]
    dest = dest_dir / fname
    _atomic_url_download(entry.url, dest)
    # A url source pointing at .xml usually has a sibling .bin: fetch it too.
    if dest.suffix == ".xml":
        bin_url = entry.url[: -len(".xml")] + ".bin"
        bin_dest = dest.with_suffix(".bin")
        try:
            _atomic_url_download(bin_url, bin_dest)
        except DownloadError:
            pass  # some IR is weight-embedded; tolerate a missing .bin
    return dest


def _fetch_by_src(entry: ModelEntry, dest_dir: Path) -> Path:
    """Dispatch to the right fetcher for ``entry.src`` (no fallback handling)."""
    if entry.src == "hf":
        return _fetch_hf(entry, dest_dir)
    if entry.src == "url":
        return _fetch_url(entry, dest_dir)
    raise DownloadError(
        f"Source type '{entry.src}' for '{entry.name}' is not handled by the "
        f"downloader (genai/anomalib delegate to their own loaders)."
    )


def _fallback_entry(entry: ModelEntry) -> ModelEntry:
    """Build a ModelEntry from ``entry.fallback``, inheriting shared metadata."""
    fb = dict(entry.fallback or {})
    return ModelEntry(
        name=entry.name,
        src=fb.get("src", "url"),
        repo=fb.get("repo"),
        filename=fb.get("filename"),
        subfolder=fb.get("subfolder"),
        url=fb.get("url"),
        sha256=fb.get("sha256"),
        license=entry.license,
        task=entry.task,
        precision=entry.precision,
    )


def fetch(entry: ModelEntry) -> Path:
    """Ensure the source artifact for ``entry`` is present locally and return it.

    Resolution order:

    1. Offline (``OVKIT_OFFLINE=1``): return a cached copy or raise.
    2. The entry's primary source.
    3. ``entry.fallback`` (e.g. the upstream original) if the primary fails —
       so a mirror outage degrades to the original host instead of breaking.
    """
    dest_dir = downloads_dir(entry.name)

    # Offline fast-path: reuse anything already downloaded.
    if is_offline():
        cached = _find_cached_source(entry, dest_dir)
        if cached is not None:
            return cached
        raise OfflineError(
            f"OVKIT_OFFLINE=1 but '{entry.name}' is not in the cache "
            f"({dest_dir}). Disable offline mode to download it."
        )

    try:
        path = _fetch_by_src(entry, dest_dir)
    except DownloadError as primary_exc:
        if not entry.fallback:
            raise
        try:
            path = _fetch_by_src(_fallback_entry(entry), dest_dir)
        except DownloadError as fb_exc:
            raise DownloadError(
                f"'{entry.name}': primary source failed ({primary_exc}); "
                f"fallback also failed ({fb_exc})."
            ) from fb_exc

    _verify(path, entry.sha256)
    return path


def _find_cached_source(entry: ModelEntry, dest_dir: Path) -> Path | None:
    if entry.filename:
        candidate = dest_dir / entry.filename
        if candidate.is_file():
            return candidate
    for pattern in ("*.xml", "*.onnx"):
        hits = sorted(dest_dir.rglob(pattern))
        if hits:
            return hits[0]
    return None
