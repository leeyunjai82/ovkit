#!/usr/bin/env python3
"""Build the ovkit model mirror on Hugging Face.

For every selected model this script:

1. downloads the source artifact (ONNX / IR) from its registered source,
2. converts it to OpenVINO IR (cached under ``$OVKIT_HOME``), and
3. uploads the IR (``model.xml`` + ``model.bin``) and the original source to a
   target Hugging Face repo, with a per-model model card.

Where the models come from
--------------------------
* ovkit's runtime manifest (``src/ovkit/manifests/*.yaml``).
* curated source lists under ``scripts/mirror_extra/*.yaml`` (DETR detectors,
  OMZ Apache face models) — loaded automatically unless ``--no-extra``.
* ``--omz-intel``: the **entire** Open Model Zoo ``intel`` set, enumerated live
  from GitHub. Only Apache-2.0 models are kept, so the mirror stays clean. This
  is how you mirror "all the OpenVINO models", not a hand-written list.

It prints, per model, the manifest entry that points ovkit at the mirror — paste
those into ``src/ovkit/manifests/models.yaml`` to switch ovkit over to it.

Run it where Hugging Face / openvinotoolkit.org / GitHub are reachable, with a
**write** token::

    export HF_TOKEN=hf_xxx                  # or: huggingface-cli login
    pip install ovkit
    python scripts/build_mirror.py --repo leeyunjai/ovkit-models --omz-intel

Flags::

    --models NAME ...     subset by name (default: every selected model)
    --omz-intel           also mirror all Apache-2.0 OMZ intel models
    --manifest PATH ...   extra source manifests to include
    --no-extra            do not auto-load scripts/mirror_extra/
    --precision fp16      IR precision (default: fp16)
    --private             create the repo private
    --dry-run             download + convert only, no upload
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
import urllib.request
from pathlib import Path

import yaml

# Reuse ovkit's own resolve/download/convert so the mirror matches what ovkit
# will later fetch from it.
from ovkit.core import registry as reg
from ovkit.core.convert import to_ir
from ovkit.core.download import fetch
from ovkit.core.registry import ModelEntry, list_models, resolve

_EXTRA_DIR = Path(__file__).resolve().parent / "mirror_extra"

# OMZ "intel" pre-trained models live in this repo; each has a model.yml with
# its source URLs, task_type and license.
_OMZ_API = (
    "https://api.github.com/repos/openvinotoolkit/open_model_zoo/contents/models/intel?ref=master"
)
_OMZ_RAW = (
    "https://raw.githubusercontent.com/openvinotoolkit/open_model_zoo/master/models/intel/"
    "{name}/model.yml"
)

# OMZ task_type -> ovkit task (folder name in the mirror).
_TASK_MAP = {
    "detection": "detect",
    "classification": "classify",
    "semantic_segmentation": "segment",
    "instance_segmentation": "segment",
    "human_pose_estimation": "pose",
    "face_recognition": "face",
    "facial_landmarks_regression": "face",
    "head_pose_estimation": "face",
    "object_attributes": "classify",
}


# --- source selection ------------------------------------------------------


def _load_extra_manifests(extra_paths: list[str], use_dir: bool) -> None:
    """Make curated/extra source manifests visible to the ovkit registry."""
    paths: list[str] = []
    if use_dir and _EXTRA_DIR.is_dir():
        paths.append(str(_EXTRA_DIR))
    paths.extend(extra_paths)
    if not paths:
        return
    existing = os.environ.get("OVKIT_MANIFESTS", "")
    os.environ["OVKIT_MANIFESTS"] = os.pathsep.join([*paths, existing]).strip(os.pathsep)
    reg.reload()


def _http_json(url: str) -> object:
    headers = {"User-Agent": "ovkit-mirror", "Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token and not token.isascii():
        # A non-ASCII value (e.g. a placeholder) breaks HTTP header encoding;
        # ignore it and fall back to unauthenticated (still fine for one call).
        print("  warning: GITHUB_TOKEN has non-ASCII characters; ignoring it.")
        token = None
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, headers=headers)  # noqa: S310
    with urllib.request.urlopen(req) as resp:  # noqa: S310
        return json.load(resp)


def _http_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "ovkit-mirror"})  # noqa: S310
    with urllib.request.urlopen(req) as resp:  # noqa: S310
        return resp.read().decode("utf-8")


def _omz_source_url(spec: dict) -> str | None:
    """Pick the FP16 ``.xml`` source URL from an OMZ model.yml ``files`` list."""
    for f in spec.get("files", []):
        name = str(f.get("name", ""))
        if not (name.endswith(".xml") and "FP16" in name):
            continue
        src = f.get("source")
        if isinstance(src, str):
            return src
        if isinstance(src, dict):
            return src.get("url") or src.get("$ref")
    return None


def omz_intel_entries() -> list[ModelEntry]:
    """Enumerate every OMZ ``intel`` model as a ModelEntry (url src).

    The entire ``models/intel`` tree is Apache-2.0 (these are OpenVINO's own
    pretrained models). The per-model ``license`` field in ``model.yml`` holds a
    URL to the repo LICENSE, not an SPDX id, so we do not SPDX-match it — we
    only read ``models/intel`` and treat all of it as Apache-2.0. (``models/
    public`` carries varied third-party licenses and is intentionally skipped.)
    """
    print("Enumerating Open Model Zoo intel models from GitHub...")
    listing = _http_json(_OMZ_API)
    entries: list[ModelEntry] = []
    no_ir = 0
    for item in listing if isinstance(listing, list) else []:
        if item.get("type") != "dir":
            continue
        name = item["name"]
        try:
            spec = yaml.safe_load(_http_text(_OMZ_RAW.format(name=name))) or {}
        except Exception:
            continue
        url = _omz_source_url(spec)
        if not url:
            no_ir += 1  # model has no FP16 IR (e.g. composite/FP32-only)
            continue
        task = _TASK_MAP.get(str(spec.get("task_type", "")), str(spec.get("task_type") or "model"))
        entries.append(
            ModelEntry(
                name=name.replace("-", "_"),
                src="url",
                url=url,
                task=task,
                precision="fp16",
                license="apache-2.0",
                license_url=spec.get("license"),
            )
        )
    print(f"  found {len(entries)} Apache-2.0 OMZ models ({no_ir} without an FP16 IR)")
    return entries


# --- upload ----------------------------------------------------------------


def _mirror_paths(entry: ModelEntry) -> tuple[str, str, str]:
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

        Files: `model.xml` / `model.bin` (OpenVINO IR), plus the original source.

        ```python
        from ovkit import Model
        model = Model("{entry.name}")
        ```
        """)


def _manifest_snippet(entry: ModelEntry, repo: str) -> str:
    """Emit a manifest entry: mirror as primary, the upstream source as fallback."""
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
    if entry.license_url:
        lines.append(f"  license_url: {entry.license_url}")
    # Fallback = the original upstream source, so a mirror outage still resolves.
    lines.append("  fallback:")
    lines.append(f"    src: {entry.src}")
    if entry.repo:
        lines.append(f"    repo: {entry.repo}")
    if entry.filename:
        lines.append(f"    filename: {entry.filename}")
    if entry.url:
        lines.append(f"    url: {entry.url}")
    return "\n".join(lines)


def _upload_license(api, entry: ModelEntry, repo: str, base: str) -> None:
    """Best-effort: place the upstream LICENSE/attribution beside the model.

    Apache-2.0/permissive redistribution requires preserving the license and
    attribution, so we fetch the original license text where we can.
    """
    text: str | None = None
    if entry.license_url:
        try:
            text = _http_text(entry.license_url)
        except Exception:
            text = None
    if text is None and entry.src == "hf" and entry.repo:
        try:
            from huggingface_hub import hf_hub_download

            for cand in ("LICENSE", "LICENSE.txt", "NOTICE"):
                try:
                    lp = hf_hub_download(repo_id=entry.repo, filename=cand)
                    text = Path(lp).read_text(encoding="utf-8", errors="replace")
                    break
                except Exception:
                    continue
        except Exception:
            text = None
    if text is None:
        # No license file found upstream: record the declared license + source.
        text = (
            f"{entry.name}\nLicense: {entry.license}\n"
            f"Source: {entry.repo or entry.url}\nLicense URL: {entry.license_url or 'n/a'}\n"
        )
    api.upload_file(
        path_or_fileobj=text.encode("utf-8"),
        path_in_repo=f"{base}/LICENSE",
        repo_id=repo,
        repo_type="model",
    )


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
    card = _model_card(entry, source.name).encode("utf-8")
    api.upload_file(
        path_or_fileobj=card, path_in_repo=f"{base}/README.md", repo_id=repo, repo_type="model"
    )
    _upload_license(api, entry, repo, base)
    print(f"  uploaded -> {repo}/{base}/")
    return _manifest_snippet(entry, repo)


def _upload_repo_readme(api, repo: str) -> None:
    """Write a top-level README describing the mirror and its license policy."""
    body = textwrap.dedent("""\
        ---
        license: apache-2.0
        ---

        # ovkit model mirror

        OpenVINO IR models served to [ovkit](https://github.com/leeyunjai82/ovkit).
        Each model lives under `<task>/<name>/` with `model.xml` + `model.bin`,
        the original source artifact, a `README.md` model card, and the upstream
        `LICENSE`.

        ## License

        This repository **redistributes** third-party models. Every model is
        permissively licensed (Apache-2.0 / MIT / BSD); the original license and
        attribution are preserved in each model's `LICENSE` file. No AGPL
        (YOLO/Ultralytics) or non-commercial (InsightFace pretrained) weights are
        hosted here. If you believe a model is mis-licensed, please open an issue.
        """)
    api.upload_file(
        path_or_fileobj=body.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo,
        repo_type="model",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the ovkit HF model mirror.")
    parser.add_argument("--repo", default="leeyunjai/ovkit-models", help="target HF repo id")
    parser.add_argument("--models", nargs="*", help="subset by name (default: all selected)")
    parser.add_argument("--omz-intel", action="store_true", help="mirror all Apache OMZ models")
    parser.add_argument("--manifest", nargs="*", default=[], help="extra source manifest paths")
    parser.add_argument("--no-extra", action="store_true", help="skip scripts/mirror_extra/")
    parser.add_argument("--precision", default="fp16", help="IR precision (default: fp16)")
    parser.add_argument("--private", action="store_true", help="create the repo as private")
    parser.add_argument("--dry-run", action="store_true", help="download + convert, no upload")
    args = parser.parse_args(argv)

    _load_extra_manifests(args.manifest, use_dir=not args.no_extra)

    # Gather the candidate entries (manifest + extras), de-duplicated by name.
    selected: dict[str, ModelEntry] = {}
    names = args.models or list_models()
    for name in names:
        entry = resolve(name)  # also enforces the permissive-license policy
        if entry is None:
            print(f"warning: '{name}' is not registered; skipping.", file=sys.stderr)
            continue
        selected[entry.name] = entry

    if args.omz_intel:
        try:
            for entry in omz_intel_entries():
                if args.models and entry.name not in args.models:
                    continue
                selected.setdefault(entry.name, entry)
        except Exception as exc:
            print(f"warning: OMZ enumeration failed: {exc}", file=sys.stderr)

    entries = list(selected.values())
    if not entries:
        print("No models to mirror.", file=sys.stderr)
        return 1
    print(f"Selected {len(entries)} model(s) to mirror -> {args.repo}")

    api = None
    if not args.dry_run:
        try:
            from huggingface_hub import HfApi
        except ImportError:
            print("huggingface_hub is required (pip install ovkit).", file=sys.stderr)
            return 1
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
        _upload_repo_readme(api, args.repo)

    snippets, failed = [], []
    for entry in entries:
        try:
            snippets.append(build_one(api, entry, args.repo, args.precision, args.dry_run))
        except Exception as exc:  # keep going across the remaining models
            failed.append(entry.name)
            print(f"  ERROR mirroring '{entry.name}': {exc}", file=sys.stderr)

    if snippets:
        print("\n" + "=" * 70)
        print("Manifest entries pointing at the mirror (paste into models.yaml):\n")
        print("\n\n".join(snippets))
    print(f"\nDone: {len(snippets)} mirrored, {len(failed)} failed.")
    if failed:
        print("Failed: " + ", ".join(failed), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
