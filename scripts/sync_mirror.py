#!/usr/bin/env python3
"""Copy every model ovkit serves into leeyunjai/ovkit-models.

Run this where Hugging Face is reachable and ``HF_TOKEN`` has write access::

    export HF_TOKEN=hf_...
    python scripts/sync_mirror.py            # check what is missing
    python scripts/sync_mirror.py --upload   # copy it in

**One mirror, on purpose.** ovkit downloads from exactly one repository, so a
school can point an offline machine at a single copy, and so keeping ovkit
alive means keeping one repository alive. Models that originate elsewhere are
copied here; their home repository stays registered as the fallback.

Two sources feed it today:

``leeyunjai/rtdetr``
    The RT-DETR detectors, from the project that trains them. Apache-2.0 code
    and weights.

``leeyunjai/edge-lab``
    Depth Anything V2 **Small** and U2-Net. That repository is tagged
    **agpl-3.0** while both of these models are individually Apache-2.0 —
    copying them here is what keeps ovkit free of copyleft, the same rule that
    put RT-DETR in it instead of YOLO. Only the Small depth checkpoint may be
    taken: Base, Large and Giant are CC-BY-NC-4.0.

Everything else in edge-lab is deliberately left alone — the OMZ face suite
and super-resolution are already mirrored, and the MediaPipe ``.task`` bundles
and EasyOCR ``.pth`` checkpoints are not formats OpenVINO reads.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

TARGET_REPO = "leeyunjai/ovkit-models"

#: (source repo, source path) -> (path in the mirror, what it is, its licence)
FILES: dict[tuple[str, str], tuple[str, str, str]] = {}

for _variant in ("r18", "r34", "r50"):
    for _ext in ("xml", "bin"):
        FILES[("leeyunjai/rtdetr", f"rtdetr-{_variant}/rtdetr-{_variant}.{_ext}")] = (
            f"detect/rtdetr_{_variant}/model.{_ext}",
            f"RT-DETR {_variant.upper()} (COCO-80)",
            "apache-2.0",
        )
    FILES[("leeyunjai/rtdetr", f"rtdetr-{_variant}/labels.txt")] = (
        f"detect/rtdetr_{_variant}/labels.txt",
        f"RT-DETR {_variant.upper()} class names",
        "apache-2.0",
    )

FILES[("leeyunjai/edge-lab", "gan/depth-v2s.xml")] = (
    "depth/depth_anything_v2_small/model.xml",
    "Depth Anything V2 Small",
    "apache-2.0",
)
FILES[("leeyunjai/edge-lab", "gan/depth-v2s.bin")] = (
    "depth/depth_anything_v2_small/model.bin",
    "Depth Anything V2 Small",
    "apache-2.0",
)
FILES[("leeyunjai/edge-lab", "gan/u2net.onnx")] = (
    "background/u2net/u2net.onnx",
    "U2-Net",
    "apache-2.0",
)

LICENSE_NOTE = """\
{what}

Mirrored into ovkit-models from {source}. ovkit downloads from one repository
so an offline site has one copy to keep; the source repository above remains
the upstream. This model is {licence}.

Depth Anything V2: only the **Small** checkpoint is Apache-2.0 and may be
mirrored here. Base, Large and Giant are CC-BY-NC-4.0.
"""

#: Files whose absence upstream is not a failure (a model may ship no labels).
OPTIONAL = {"labels.txt"}


def _is_optional(source_path: str) -> bool:
    return Path(source_path).name in OPTIONAL


def check() -> tuple[int, list[tuple[str, str]]]:
    """Report what is missing from the mirror; return (exit code, to-copy)."""
    from huggingface_hub import list_repo_files

    try:
        already = set(list_repo_files(TARGET_REPO))
    except Exception as exc:
        print(f"could not list {TARGET_REPO}: {exc}", file=sys.stderr)
        return 2, []

    listings: dict[str, set[str]] = {}
    todo: list[tuple[str, str]] = []
    missing_upstream: list[str] = []

    for (repo, source_path), (dest, what, licence) in FILES.items():
        if repo not in listings:
            try:
                listings[repo] = set(list_repo_files(repo))
            except Exception as exc:
                print(f"could not list {repo}: {exc}", file=sys.stderr)
                return 2, []
        if source_path not in listings[repo]:
            if not _is_optional(source_path):
                missing_upstream.append(f"{repo}/{source_path}")
                print(f"  [MISSING] {repo}/{source_path}")
            continue
        if dest in already:
            print(f"  [have   ] {dest}")
            continue
        print(f"  [copy   ] {repo}/{source_path:34s} -> {dest}   ({what}, {licence})")
        todo.append((repo, source_path))

    if missing_upstream:
        print(f"\n{len(missing_upstream)} source file(s) not found upstream.", file=sys.stderr)
        return 1, []
    if not todo:
        print("\nThe mirror already has everything.")
        return 0, []
    print(f"\n{len(todo)} file(s) to copy. Re-run with --upload.")
    return 0, todo


def upload(todo: list[tuple[str, str]]) -> int:
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    operations = []
    notes: set[str] = set()
    for repo, source_path in todo:
        dest, what, licence = FILES[(repo, source_path)]
        print(f"fetching {repo}/{source_path} ...")
        local = hf_hub_download(repo_id=repo, filename=source_path)
        operations.append(CommitOperationAdd(path_in_repo=dest, path_or_fileobj=local))

        note = str(Path(dest).parent / "LICENSE.md")
        if note not in notes:
            notes.add(note)
            operations.append(
                CommitOperationAdd(
                    path_in_repo=note,
                    path_or_fileobj=LICENSE_NOTE.format(
                        what=what, source=repo, licence=licence
                    ).encode("utf-8"),
                )
            )

    print(f"\ncommitting {len(operations)} file(s) to {TARGET_REPO} ...")
    HfApi().create_commit(
        repo_id=TARGET_REPO,
        operations=operations,
        commit_message="Mirror RT-DETR, Depth Anything V2 Small and U2-Net",
    )
    print("done. ovkit needs no further changes:")
    print("  python -c \"from ovkit import Model; print(Model('detect', 'photo.jpg'))\"")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--upload", action="store_true", help="copy (default: check only)")
    args = parser.parse_args()
    status, todo = check()
    if status or not args.upload:
        return status
    return upload(todo)


if __name__ == "__main__":
    raise SystemExit(main())
