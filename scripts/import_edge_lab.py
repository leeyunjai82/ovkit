#!/usr/bin/env python3
"""Copy two models from leeyunjai/edge-lab into the ovkit mirror.

Run this where Hugging Face is reachable and ``HF_TOKEN`` has write access::

    export HF_TOKEN=hf_...
    python scripts/import_edge_lab.py            # check first
    python scripts/import_edge_lab.py --upload

Why a copy rather than pointing ovkit at edge-lab: that repository is tagged
**agpl-3.0**, and ovkit's whole licence policy — the reason it uses RT-DETR
instead of YOLO — is that nothing AGPL ships with it. The two models below are
individually Apache-2.0, so they belong in the mirror whose licence story is
clean. Everything else in edge-lab is either already mirrored (the OMZ face
suite, super-resolution) or not in a form OpenVINO reads (MediaPipe ``.task``
bundles, EasyOCR ``.pth`` checkpoints).

    gan/depth-v2s.xml + .bin   Depth Anything V2 **Small** — Apache-2.0.
                               Base/Large/Giant are CC-BY-NC and must not be
                               mirrored; only the small one may.
    gan/u2net.onnx             U2-Net salient-object segmentation — Apache-2.0.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SOURCE_REPO = "leeyunjai/edge-lab"
TARGET_REPO = "leeyunjai/ovkit-models"

#: source path -> (path in the ovkit mirror, what it is, its own licence)
FILES: dict[str, tuple[str, str, str]] = {
    "gan/depth-v2s.xml": ("depth/depth_anything_v2_small/model.xml", "Depth Anything V2 Small", "apache-2.0"),
    "gan/depth-v2s.bin": ("depth/depth_anything_v2_small/model.bin", "Depth Anything V2 Small", "apache-2.0"),
    "gan/u2net.onnx": ("background/u2net/u2net.onnx", "U2-Net", "apache-2.0"),
}

LICENSE_NOTE = """\
{what}

Mirrored into ovkit-models from {source} so that ovkit ships nothing under a
copyleft or non-commercial licence. This model is {licence}.

Depth Anything V2: only the **Small** checkpoint is Apache-2.0. The Base,
Large and Giant checkpoints are CC-BY-NC-4.0 and must never be mirrored here.
"""


def check() -> int:
    """Report what would be copied, and whether the sources are really there."""
    from huggingface_hub import list_repo_files

    print(f"source: {SOURCE_REPO}\ntarget: {TARGET_REPO}\n")
    try:
        present = set(list_repo_files(SOURCE_REPO))
    except Exception as exc:
        print(f"could not list {SOURCE_REPO}: {exc}", file=sys.stderr)
        return 2

    missing = [src for src in FILES if src not in present]
    for src, (dest, what, licence) in FILES.items():
        mark = "MISSING" if src in missing else "ok"
        print(f"  [{mark:7s}] {src:24s} -> {dest}   ({what}, {licence})")
    if missing:
        print(f"\n{len(missing)} source file(s) not found — nothing was uploaded.", file=sys.stderr)
        return 1
    print("\nAll sources present. Re-run with --upload to copy them.")
    return 0


def upload() -> int:
    """Download each file and commit it into the ovkit mirror."""
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

    api = HfApi()
    operations = []
    for src, (dest, what, licence) in FILES.items():
        print(f"fetching {src} ...")
        local = hf_hub_download(repo_id=SOURCE_REPO, filename=src)
        operations.append(CommitOperationAdd(path_in_repo=dest, path_or_fileobj=local))
        note = Path(dest).parent / "LICENSE.md"
        operations.append(
            CommitOperationAdd(
                path_in_repo=str(note),
                path_or_fileobj=LICENSE_NOTE.format(
                    what=what, source=SOURCE_REPO, licence=licence
                ).encode("utf-8"),
            )
        )

    print(f"\ncommitting {len(operations)} file(s) to {TARGET_REPO} ...")
    api.create_commit(
        repo_id=TARGET_REPO,
        operations=operations,
        commit_message="Add Depth Anything V2 Small and U2-Net (both Apache-2.0)",
    )
    print("done. ovkit picks them up with no further changes:")
    print('  python -c "from ovkit import Model; print(Model(\'depth\', \'photo.jpg\'))"')
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--upload", action="store_true", help="actually copy (default: check only)"
    )
    args = parser.parse_args()
    if not args.upload:
        return check()
    status = check()
    return status if status else upload()


if __name__ == "__main__":
    raise SystemExit(main())
