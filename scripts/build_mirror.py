#!/usr/bin/env python3
"""Build the ovkit model mirror on Hugging Face.

For every model in the ovkit manifest this script:

1. downloads the source artifact (ONNX / IR) from its registered source,
2. converts it to OpenVINO IR (cached under ``$OVKIT_HOME``), and
3. uploads the IR (``model.xml`` + ``model.bin``) and the original source to a
   target Hugging Face repo, with a per-model model card.

It also prints, for each model, the manifest entry that points ovkit at the
mirror — paste those into ``src/ovkit/manifests/models.yaml`` to switch ovkit
over to your mirror.

Run it where Hugging Face is reachable (e.g. your laptop), authenticated with a
**write** token::

    export HF_TOKEN=hf_xxx                 # or: huggingface-cli login
    pip install ovkit                      # provides openvino + huggingface_hub
    python scripts/build_mirror.py --repo leeyunjai/ovkit-models

Useful flags::

    --models rtdetr_r50 rtdetrv2_r18       # subset (default: all in manifest)
    --precision fp16                       # IR precision (default: fp16)
    --private                              # create the repo private
    --dry-run                              # download + convert only, no upload

Only permissive-licensed models are accepted (the manifest enforces this), so
the mirror stays Apache-2.0/MIT/BSD clean.
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap

# Reuse ovkit's own resolve/download/convert so the mirror matches what ovkit
# will later fetch from it.
from ovkit.core.convert import to_ir
from ovkit.core.download import fetch
from ovkit.core.registry import ModelEntry, list_models, resolve


def _mirror_paths(entry: ModelEntry) -> tuple[str, str, str]:
    """Return repo-relative ``(xml, bin, source)`` paths for a model."""
    base = f"{entry.task or 'model'}/{entry.name}"
    return f"{base}/model.xml", f"{base}/model.bin", base


def _model_card(entry: ModelEntry, source_name: str) -> str:
    return textwrap.dedent(f"""\
        ---
        license: {entry.license}
        tags:
          - openvino
          - ovkit
          - {entry.task}
        ---

        # {entry.name}

        OpenVINO IR mirror for [ovkit](https://github.com/leeyunjai82/ovkit).

        | field      | value |
        | ---------- | ----- |
        | task       | `{entry.task}` |
        | license    | `{entry.license}` |
        | precision  | `{entry.precision}` |
        | source     | `{entry.src}` (`{entry.repo or entry.url}`) |

        Files:

        - `model.xml` / `model.bin` — OpenVINO IR (use this with ovkit)
        - `{source_name}` — original source artifact

        Load with ovkit:

        ```python
        from ovkit import Model
        model = Model("{entry.name}")
        ```
        """)


def _manifest_snippet(entry: ModelEntry, repo: str) -> str:
    xml, _, _ = _mirror_paths(entry)
    lines = [
        f"{entry.name}:",
        "  src: hf",
        f"  repo: {repo}",
        f"  filename: {xml}",
        f"  task: {entry.task}",
        f"  precision: {entry.precision}",
        f"  license: {entry.license}",
    ]
    if entry.imgsz:
        lines.append(f"  imgsz: {entry.imgsz}")
    return "\n".join(lines)


def build_one(api, entry: ModelEntry, repo: str, precision: str, dry_run: bool) -> str:
    print(f"\n=== {entry.name} ({entry.task}, {entry.license}) ===")
    source = fetch(entry)
    print(f"  source : {source}")
    ir_xml = to_ir(source, entry.name, precision)
    ir_bin = ir_xml.with_suffix(".bin")
    print(f"  IR     : {ir_xml}")

    xml_path, bin_path, base = _mirror_paths(entry)
    if dry_run:
        print("  (dry-run: skipping upload)")
        return _manifest_snippet(entry, repo)

    api.upload_file(
        path_or_fileobj=str(ir_xml), path_in_repo=xml_path, repo_id=repo, repo_type="model"
    )
    if ir_bin.is_file():
        api.upload_file(
            path_or_fileobj=str(ir_bin), path_in_repo=bin_path, repo_id=repo, repo_type="model"
        )
    api.upload_file(
        path_or_fileobj=str(source),
        path_in_repo=f"{base}/{source.name}",
        repo_id=repo,
        repo_type="model",
    )
    # Per-model card (uploaded as a README inside the model's folder).
    card = _model_card(entry, source.name).encode("utf-8")
    api.upload_file(
        path_or_fileobj=card, path_in_repo=f"{base}/README.md", repo_id=repo, repo_type="model"
    )
    print(f"  uploaded -> {repo}/{base}/")
    return _manifest_snippet(entry, repo)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the ovkit HF model mirror.")
    parser.add_argument("--repo", default="leeyunjai/ovkit-models", help="target HF repo id")
    parser.add_argument("--models", nargs="*", help="model names (default: all registered)")
    parser.add_argument("--precision", default="fp16", help="IR precision (default: fp16)")
    parser.add_argument("--private", action="store_true", help="create the repo as private")
    parser.add_argument("--dry-run", action="store_true", help="download + convert, no upload")
    args = parser.parse_args(argv)

    names = args.models or list_models()
    entries: list[ModelEntry] = []
    for name in names:
        entry = resolve(name)  # also enforces the permissive-license policy
        if entry is None:
            print(f"warning: '{name}' is not a registered model; skipping.", file=sys.stderr)
            continue
        entries.append(entry)
    if not entries:
        print("No models to mirror.", file=sys.stderr)
        return 1

    api = None
    if not args.dry_run:
        try:
            from huggingface_hub import HfApi
        except ImportError:
            print("huggingface_hub is required (pip install ovkit).", file=sys.stderr)
            return 1
        token = os.environ.get("HF_TOKEN")
        api = HfApi(token=token)
        api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
        print(f"target repo: {args.repo} (private={args.private})")

    snippets = []
    for entry in entries:
        try:
            snippets.append(build_one(api, entry, args.repo, args.precision, args.dry_run))
        except Exception as exc:  # keep going across the remaining models
            print(f"  ERROR mirroring '{entry.name}': {exc}", file=sys.stderr)

    if snippets:
        print("\n" + "=" * 70)
        print("Manifest entries pointing at the mirror (paste into models.yaml):\n")
        print("\n\n".join(snippets))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
