"""Model registry: load the manifest and resolve a name to a source spec.

The registry is intentionally data-driven. Models live in YAML manifests
(``src/ovkit/manifests/*.yaml`` plus any user-supplied paths), never hardcoded
in Python. Adding a model is a one-line YAML edit.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from .constants import is_permissive
from .errors import LicenseError

#: Directory holding the bundled manifest files.
_MANIFEST_DIR = Path(__file__).resolve().parent.parent / "manifests"

#: Extra manifest search paths from ``$OVKIT_MANIFESTS`` (os.pathsep separated).
_ENV_MANIFESTS = "OVKIT_MANIFESTS"


@dataclass(frozen=True)
class ModelEntry:
    """A single resolved manifest entry.

    Attributes mirror the YAML schema documented in ``manifests/models.yaml``.
    """

    name: str
    src: str
    task: str | None = None
    license: str | None = None
    precision: str = "fp16"
    repo: str | None = None
    filename: str | None = None
    subfolder: str | None = None
    url: str | None = None
    sha256: str | None = None
    imgsz: int | None = None
    license_url: str | None = None
    fallback: dict[str, Any] | None = None
    preprocess: dict[str, Any] = field(default_factory=dict)
    postprocess: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, name: str, data: dict[str, Any]) -> ModelEntry:
        known = {
            "src",
            "task",
            "license",
            "precision",
            "repo",
            "filename",
            "subfolder",
            "url",
            "sha256",
            "imgsz",
            "license_url",
            "fallback",
            "preprocess",
            "postprocess",
        }
        kwargs: dict[str, Any] = {k: data[k] for k in known if k in data}
        kwargs.setdefault("precision", "fp16")
        extra = {k: v for k, v in data.items() if k not in known}
        return cls(name=name, extra=extra, **kwargs)


def _manifest_paths() -> list[Path]:
    paths: list[Path] = []
    if _MANIFEST_DIR.is_dir():
        paths.extend(sorted(_MANIFEST_DIR.glob("*.yaml")))
        paths.extend(sorted(_MANIFEST_DIR.glob("*.yml")))
    env = os.environ.get(_ENV_MANIFESTS, "")
    for raw in env.split(os.pathsep):
        raw = raw.strip()
        if not raw:
            continue
        p = Path(raw).expanduser()
        if p.is_dir():
            paths.extend(sorted(p.glob("*.yaml")))
            paths.extend(sorted(p.glob("*.yml")))
        elif p.is_file():
            paths.append(p)
    return paths


@lru_cache(maxsize=1)
def _load_raw() -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for path in _manifest_paths():
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:  # pragma: no cover - corrupt manifest
            raise OVKitManifestError(f"Failed to parse manifest {path}: {exc}") from exc
        if not isinstance(data, dict):
            continue
        for name, spec in data.items():
            if isinstance(spec, dict):
                merged[name] = spec  # later manifests override earlier ones
    return merged


class OVKitManifestError(Exception):
    """Raised when a manifest file cannot be parsed."""


def reload() -> None:
    """Clear the cached manifest (call after editing manifests at runtime)."""
    _load_raw.cache_clear()


def list_models() -> list[str]:
    """Return all registered model names, sorted."""
    return sorted(_load_raw().keys())


def resolve(name: str) -> ModelEntry | None:
    """Resolve a registered model ``name`` to a :class:`ModelEntry`.

    Returns ``None`` if the name is not in any manifest. The entry's license is
    validated to be permissive; non-permissive entries raise
    :class:`~ovkit.core.errors.LicenseError` so they can never load.
    """
    raw = _load_raw().get(name)
    if raw is None:
        return None
    entry = ModelEntry.from_dict(name, raw)
    if not is_permissive(entry.license):
        raise LicenseError(
            f"Model '{name}' declares license '{entry.license}', which is not on "
            f"ovkit's permissive allow-list. Only permissive (Apache-2.0/MIT/BSD/...) "
            f"models may be registered. Refusing to load."
        )
    return entry
