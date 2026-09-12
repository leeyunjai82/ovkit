#!/usr/bin/env python3
"""Find files in leeyunjai/ovkit-models that no ovkit manifest asks for.

The mirror is the one place ovkit downloads from, so it accumulates: models
that were curated out of the lineup, weights uploaded under an old path, a
precision nobody ends up serving. Those are not free — the mirror is what a
school clones onto an offline machine, and what has to stay alive for ovkit to
keep working.

    python scripts/audit_mirror.py           # list what nothing references
    python scripts/audit_mirror.py --prune   # delete it (needs a write token)

Or run it from Actions -> "Sync the model mirror", which holds the token.

What counts as referenced
-------------------------
Every manifest entry (and every ``fallback``) that points at the mirror keeps:

* its ``filename``, plus the ``.bin`` weights and ``labels.txt`` that
  :func:`ovkit.core.download` fetches alongside an ``.xml``;
* everything under its ``subfolder``, for the genai pipelines that are
  downloaded as a whole directory;
* the ``LICENSE.md`` provenance note next to it.

Read the list before pruning. A model that is *deliberately* kept in the
mirror without being in a manifest — a checkpoint someone downloads by hand,
a spare precision — will show up here as an orphan; add it to KEEP below
rather than re-uploading it after the fact.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

TARGET_REPO = "leeyunjai/ovkit-models"

MANIFEST_DIR = Path(__file__).resolve().parent.parent / "src" / "ovkit" / "manifests"

#: Repository furniture at the root, kept whatever the manifests say.
KEEP: tuple[str, ...] = (
    ".gitattributes",
    ".gitignore",
    "README.md",
    "LICENSE",
    "LICENSE.md",
    "LICENSE.txt",
)

#: Provenance that travels with a model: kept when its folder holds a model
#: ovkit still serves, dropped with the folder when it does not. Where a model
#: came from and under what licence is not furniture to tidy away.
COMPANIONS = frozenset({"README.md", "LICENSE", "LICENSE.md", "LICENSE.txt", "labels.txt"})


def _mirror_specs() -> list[dict]:
    """Every manifest dict that points at the mirror, fallbacks included.

    Walks the parsed YAML rather than going through ``ovkit.core.registry``:
    the audit then needs only PyYAML (no openvino, no numpy) and picks up any
    new manifest without being taught about it.
    """
    specs: list[dict] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            if node.get("repo") == TARGET_REPO:
                specs.append(node)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    for path in sorted(MANIFEST_DIR.glob("*.yaml")):
        walk(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
    return specs


def referenced() -> tuple[set[str], set[str]]:
    """Return (exact paths, subtree prefixes) the manifests ask the mirror for."""
    files: set[str] = set()
    prefixes: set[str] = set()

    for spec in _mirror_specs():
        subfolder = spec.get("subfolder")
        filename = spec.get("filename")
        if filename:
            path = f"{subfolder.rstrip('/')}/{filename}" if subfolder else filename
            files.add(path)
            if path.endswith(".xml"):
                # OpenVINO IR is two files; the weights are not optional.
                files.add(path[: -len(".xml")] + ".bin")
        elif subfolder:
            # Downloaded as a whole directory (the genai pipelines).
            prefixes.add(subfolder.rstrip("/") + "/")
    return files, prefixes


def _folders(files: set[str]) -> set[str]:
    """Folders holding at least one file a manifest names."""
    return {path.rsplit("/", 1)[0] for path in files if "/" in path}


def _kept(path: str, files: set[str], prefixes: set[str]) -> bool:
    if path in files or path in KEEP:
        return True
    if any(path.startswith(prefix) for prefix in prefixes):
        return True
    name = Path(path).name
    if name in COMPANIONS and "/" in path:
        # Provenance and class names ride along with a model that is still
        # served — and go with the folder when the model itself is gone.
        return path.rsplit("/", 1)[0] in _folders(files)
    return False


def _human(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:,.0f} {unit}" if unit == "B" else f"{value:,.1f} {unit}"
        value /= 1024
    return f"{value:,.1f} GB"



def _report(orphans: list[tuple[str, int]], files: set[str]) -> None:
    """Print the orphans as two piles, because they are two different decisions.

    A mirror this size lists six hundred files; nobody reads six hundred lines
    before deciding what to delete. And the two piles are not alike:

    *Whole models* — nothing in the folder is in a manifest, so ovkit cannot
    load them at all. Deleting one removes a model from the mirror for good
    (the Hub keeps history, but nothing in ovkit points at it either way).

    *Leftovers* — the folder holds a model ovkit still serves, and these files
    sit next to it unused: the weights under their original Open Model Zoo
    name, kept when the mirror standardised on ``model.xml``. Nothing reads
    them, and they are pure duplication.
    """
    live = _folders(files)
    dead: dict[str, tuple[int, int]] = {}
    leftover: dict[str, list[tuple[str, int]]] = {}

    for path, size in orphans:
        folder = path.rsplit("/", 1)[0] if "/" in path else "(root)"
        if folder in live:
            leftover.setdefault(folder, []).append((path, size))
        else:
            count, total = dead.get(folder, (0, 0))
            dead[folder] = (count + 1, total + size)

    if leftover:
        total = sum(size for rows in leftover.values() for _, size in rows)
        count = sum(len(rows) for rows in leftover.values())
        print(f"Beside models ovkit still serves — {count} unused file(s), {_human(total)}:\n")
        names: dict[str, int] = {}
        for rows in leftover.values():
            for path, size in rows:
                names[Path(path).name] = names.get(Path(path).name, 0) + 1
        for name, seen in sorted(names.items(), key=lambda kv: -kv[1])[:12]:
            print(f"  x{seen:<4} {name}")
        print("\n  (the model itself stays; only these extra files go)\n")

    if dead:
        total = sum(size for _, size in dead.values())
        print(f"Models no manifest mentions — {len(dead)} folder(s), {_human(total)}:\n")
        ordered = sorted(dead.items(), key=lambda kv: (-kv[1][1], kv[0]))
        for folder, (count, size) in ordered[:40]:
            print(f"  {_human(size):>10}  {folder}/  ({count} file{'s' if count > 1 else ''})")
        if len(ordered) > 40:
            rest = sum(size for _, (_, size) in ordered[40:])
            print(f"  ... and {len(ordered) - 40} more folder(s), {_human(rest)}")

        by_top: dict[str, tuple[int, int]] = {}
        for folder, (count, size) in dead.items():
            top = folder.split("/", 1)[0]
            seen, bytes_ = by_top.get(top, (0, 0))
            by_top[top] = (seen + 1, bytes_ + size)
        print("\n  by task:")
        for top, (seen, size) in sorted(by_top.items(), key=lambda kv: -kv[1][1]):
            print(f"    {_human(size):>10}  {top:<32} {seen} model(s)")

    print(f"\n{_human(sum(size for _, size in orphans))} in total would be freed.")
    print("Read the list, then re-run with --prune to delete them.")


def split(orphans: list[tuple[str, int]], files: set[str]) -> dict[str, list[tuple[str, int]]]:
    """Sort the orphans into the two piles ``--prune`` can act on separately."""
    live = _folders(files)
    piles: dict[str, list[tuple[str, int]]] = {"leftovers": [], "models": []}
    for path, size in orphans:
        folder = path.rsplit("/", 1)[0] if "/" in path else ""
        piles["leftovers" if folder in live else "models"].append((path, size))
    return piles


def audit() -> tuple[int, list[tuple[str, int]], set[str]]:
    """List the mirror, report orphans; return (exit code, orphans, referenced)."""
    from huggingface_hub import HfApi

    files, prefixes = referenced()
    if not files and not prefixes:
        print(
            "no manifest entry points at the mirror — refusing to call every "
            "file an orphan. Run this from a checkout of ovkit.",
            file=sys.stderr,
        )
        return 2, [], files

    try:
        tree = [
            item
            for item in HfApi().list_repo_tree(TARGET_REPO, recursive=True)
            if getattr(item, "size", None) is not None
        ]
    except Exception as exc:
        print(f"could not list {TARGET_REPO}: {exc}", file=sys.stderr)
        return 2, [], files

    orphans: list[tuple[str, int]] = []
    kept_bytes = 0
    for item in tree:
        size = int(item.size or 0)
        if _kept(item.path, files, prefixes):
            kept_bytes += size
        else:
            orphans.append((item.path, size))

    present = {item.path for item in tree}
    missing = sorted(path for path in files if path not in present)
    print(f"{TARGET_REPO}: {len(tree)} file(s), {_human(kept_bytes)} of them in use\n")

    if not orphans:
        print("Nothing to delete — every file is referenced by a manifest.")
    else:
        _report(orphans, files)

    if missing:
        print(f"\n{len(missing)} file(s) a manifest asks for but the mirror lacks:")
        for path in missing:
            print(f"  {path}")
        print("Run scripts/sync_mirror.py --upload to fill them in.")
    return 0, orphans, files


def prune(orphans: list[tuple[str, int]]) -> int:
    import os

    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        print(
            "HF_TOKEN is not set — deleting needs a token with write access to\n"
            f"{TARGET_REPO}. Either export it here, or run the\n"
            '"Sync the model mirror" workflow on GitHub, where it lives as a secret.',
            file=sys.stderr,
        )
        return 2

    from huggingface_hub import CommitOperationDelete, HfApi

    print(f"\ndeleting {len(orphans)} file(s) from {TARGET_REPO} ...")
    HfApi().create_commit(
        repo_id=TARGET_REPO,
        operations=[CommitOperationDelete(path_in_repo=path) for path, _ in orphans],
        commit_message="Drop files no ovkit manifest references",
    )
    print("done. Git history keeps them, so this is recoverable.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--prune",
        nargs="?",
        const="all",
        choices=("all", "leftovers", "models"),
        help="delete the orphans (default: list them). 'leftovers' deletes only "
        "the unused files beside models ovkit still serves — the safe half; "
        "'models' only the folders no manifest mentions.",
    )
    args = parser.parse_args()
    status, orphans, files = audit()
    if status or not args.prune or not orphans:
        return status
    if args.prune != "all":
        orphans = split(orphans, files)[args.prune]
        if not orphans:
            print(f"\nnothing in the '{args.prune}' pile.")
            return 0
    return prune(orphans)


if __name__ == "__main__":
    raise SystemExit(main())
