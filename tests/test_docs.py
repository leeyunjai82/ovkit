"""The docs make checkable claims; check them.

Both doc trees (``docs/`` and ``docs/ko/``) name models and capabilities in
tables. Those tables are written by hand and read as promises — one of them
said ``detect`` loads ``rtdetr_r50`` long after the alias had moved to
``rtdetr_r18``. A reader has no way to catch that; a test does.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from ovkit import list_pipelines
from ovkit.core import registry

DOCS = Path(__file__).resolve().parent.parent / "docs"

#: ``| `alias` | `target` | ...`` rows in the model-catalog alias tables.
_ALIAS_ROW = re.compile(r"^\|\s*`([\w./-]+)`\s*\|\s*`([\w./-]+)`\s*\|", re.M)

#: The first column of the capability tables in the feature pages.
_NAME_CELL = re.compile(r"^\|\s*`([\w_]+)`\s*\|", re.M)


def _pages(name: str) -> list[Path]:
    return [DOCS / name, DOCS / "ko" / name]


@pytest.mark.parametrize("page", _pages("models.md"), ids=lambda p: str(p.parent.name))
def test_the_alias_table_matches_the_registry(page: Path) -> None:
    """Every `alias -> model` row must be what the manifest actually resolves to."""
    rows = [
        (alias, target)
        for alias, target in _ALIAS_ROW.findall(page.read_text(encoding="utf-8"))
        if registry.resolve(alias) is not None
    ]
    assert rows, f"no alias rows found in {page} — did the table format change?"

    wrong = [
        (alias, target, registry.resolve(alias).name)
        for alias, target in rows
        if registry.resolve(alias).name != target
    ]
    assert not wrong, "\n".join(
        f"{page.parent.name}/models.md says {a} -> {claimed}, registry says {actual}"
        for a, claimed, actual in wrong
    )


@pytest.mark.parametrize("page", _pages("features.md"), ids=lambda p: str(p.parent.name))
def test_every_name_the_feature_page_lists_is_loadable(page: Path) -> None:
    """A feature page that names something ovkit cannot load is worse than none."""
    caps = set(list_pipelines())
    unknown = [
        name
        for name in _NAME_CELL.findall(page.read_text(encoding="utf-8"))
        if not name.isupper()  # the runtime table lists env vars, not models
        and name not in caps
        and registry.resolve(name) is None
    ]
    assert not unknown, f"{page} names what ovkit cannot load: {unknown}"


@pytest.mark.parametrize("page", _pages("features.md"), ids=lambda p: str(p.parent.name))
def test_the_feature_page_is_in_the_toctree(page: Path) -> None:
    """An unlisted page builds into nothing anyone can reach."""
    index = (page.parent / "index.md").read_text(encoding="utf-8")
    assert "\nfeatures\n" in index, f"{page.parent.name}/index.md does not list it"


def test_both_doc_trees_hold_the_same_pages() -> None:
    """ko/ mirrors the English tree page for page; a one-sided page goes missing."""
    en = {p.name for p in DOCS.glob("*.md")} | {p.name for p in DOCS.glob("*.rst")}
    ko = {p.name for p in (DOCS / "ko").glob("*.md")} | {
        p.name for p in (DOCS / "ko").glob("*.rst")
    }
    assert en == ko, f"only in English: {en - ko} · only in Korean: {ko - en}"


def test_no_example_calls_imshow_directly():
    """`cv2.imshow` raises on the OpenCV ovkit depends on.

    Every webcam example used it, so the demos in the README crashed for
    anyone who installed exactly what the README told them to. `Results.show()`
    is the one place that knows how to degrade.
    """
    examples = DOCS.parent / "examples"
    offenders = [
        p.name
        for p in sorted(examples.glob("*.py"))
        if "cv2.imshow" in p.read_text(encoding="utf-8")
    ]
    assert not offenders, f"use r.show(...) instead of cv2.imshow: {offenders}"
