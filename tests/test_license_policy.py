"""Which licences ovkit will load, and on what condition.

ovkit ships models to schools, so the licence check runs before anything else
— a model ovkit cannot pass on is a model ovkit will not load, however good it
is. The allow-list keeps out the two things that actually bite: AGPL, which
reaches into ovkit's own licensing, and CC-BY-NC, which forbids commercial use.

OpenRAIL-M is neither. It grants IP rights like a permissive licence and adds
use-based restrictions that **every downstream copy must carry**. ovkit serves
it only when the entry can point at the licence, because an entry that cannot
point at it cannot pass it on.
"""

from __future__ import annotations

import pytest

from ovkit.core.constants import is_permissive, is_restricted
from ovkit.core.errors import LicenseError
from ovkit.core.registry import ModelEntry, resolve


@pytest.mark.parametrize("licence", ["apache-2.0", "mit", "bsd-3-clause", "MIT", " Apache-2.0 "])
def test_permissive_ids_are_permissive(licence):
    assert is_permissive(licence)


@pytest.mark.parametrize("licence", ["agpl-3.0", "cc-by-nc-4.0", "openrail", "", None])
def test_everything_else_is_not(licence):
    assert not is_permissive(licence)


@pytest.mark.parametrize("licence", ["openrail", "openrail-m", "bigscience-openrail-m"])
def test_openrail_is_restricted_not_refused(licence):
    assert is_restricted(licence) and not is_permissive(licence)


def test_the_exception_stays_narrow():
    """It was decided for one model; a family must not ride in behind it."""
    assert not is_restricted("creativeml-openrail-m"), "that is Stable Diffusion's"
    assert not is_restricted("cc-by-nc-4.0")
    assert not is_restricted("agpl-3.0")


def _register(monkeypatch, name: str, spec: dict) -> None:
    from ovkit.core import registry

    monkeypatch.setattr(registry, "_load_raw", lambda: {name: spec})


def test_a_restricted_model_loads_when_it_carries_its_licence(monkeypatch):
    _register(
        monkeypatch,
        "spoken",
        {
            "src": "hf",
            "repo": "leeyunjai/ovkit-models",
            "filename": "tts/x/model.xml",
            "license": "openrail",
            "license_url": "https://huggingface.co/Supertone/supertonic/raw/main/LICENSE",
        },
    )
    entry = resolve("spoken")
    assert isinstance(entry, ModelEntry) and entry.license == "openrail"


def test_a_restricted_model_without_its_licence_does_not_load(monkeypatch):
    _register(
        monkeypatch,
        "spoken",
        {
            "src": "hf",
            "repo": "leeyunjai/ovkit-models",
            "filename": "tts/x/model.xml",
            "license": "openrail",
        },
    )
    with pytest.raises(LicenseError, match="pass on the same use restrictions"):
        resolve("spoken")


def test_agpl_still_refuses(monkeypatch):
    _register(monkeypatch, "nope", {"src": "hf", "license": "agpl-3.0"})
    with pytest.raises(LicenseError, match="permissive allow-list"):
        resolve("nope")
