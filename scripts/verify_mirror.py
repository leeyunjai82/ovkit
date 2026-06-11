#!/usr/bin/env python3
"""Verify the ovkit HF model mirror.

Lists every file in the mirror repo and checks that each ``<task>/<name>/`` has
a complete OpenVINO IR (``model.xml`` + ``model.bin``) and a preserved
``LICENSE``. Reports per-task counts and anything incomplete.

Run where Hugging Face is reachable::

    python scripts/verify_mirror.py --repo leeyunjai/ovkit-models

Optionally also do a live load+infer of one model straight from the mirror::

    python scripts/verify_mirror.py --smoke rtdetr_r50
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict


def verify(repo: str) -> int:
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("huggingface_hub is required (pip install ovkit).", file=sys.stderr)
        return 2

    api = HfApi()
    files = api.list_repo_files(repo, repo_type="model")

    # Group files under <task>/<name>/...
    models: dict[tuple[str, str], set[str]] = defaultdict(set)
    for f in files:
        parts = f.split("/")
        if len(parts) >= 3:
            models[(parts[0], parts[1])].add("/".join(parts[2:]))

    per_task: dict[str, int] = defaultdict(int)
    incomplete: list[tuple[str, str, list[str]]] = []
    no_license: list[str] = []
    for (task, name), fileset in sorted(models.items()):
        per_task[task] += 1
        has_xml = any(x.endswith("model.xml") for x in fileset)
        has_bin = any(x.endswith("model.bin") for x in fileset)
        if not (has_xml and has_bin):
            incomplete.append((task, name, sorted(fileset)))
        if "LICENSE" not in fileset:
            no_license.append(f"{task}/{name}")

    print(f"Mirror: {repo}")
    print(f"Total models: {len(models)}")
    for t, c in sorted(per_task.items()):
        print(f"  {t:10s}: {c}")
    print(f"\nIncomplete (missing model.xml/model.bin): {len(incomplete)}")
    for task, name, fs in incomplete:
        print(f"  ! {task}/{name}: {fs}")
    print(f"Missing LICENSE: {len(no_license)}")
    for m in no_license:
        print(f"  ~ {m}")

    ok = not incomplete
    print("\nRESULT:", "OK ✅" if ok else "INCOMPLETE ❌")
    return 0 if ok else 1


def smoke(name: str) -> int:
    """Load one model straight from the mirror and run a dummy inference."""
    import numpy as np

    from ovkit import Model

    print(f"Loading '{name}' from the mirror (this downloads + loads the IR)...")
    model = Model(name)
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    results = model(img)
    r = results[0]
    n = len(r.boxes) if r.boxes is not None else (1 if r.probs is not None else 0)
    print(f"  task={model.task}  items={n}  -> load + inference OK ✅")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Verify the ovkit HF model mirror.")
    ap.add_argument("--repo", default="leeyunjai/ovkit-models", help="mirror repo id")
    ap.add_argument("--smoke", metavar="NAME", help="also load+infer this model from the mirror")
    args = ap.parse_args(argv)

    rc = verify(args.repo)
    if args.smoke:
        print()
        rc = smoke(args.smoke) or rc
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
