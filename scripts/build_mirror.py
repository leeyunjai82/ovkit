#!/usr/bin/env python3
"""Build the ovkit model mirror on Hugging Face.

For every selected model this script:

1. downloads the source artifact (ONNX / IR) from its registered source,
2. converts it to OpenVINO IR (cached under ``$OVKIT_HOME``), and
3. uploads the IR (``model.xml`` + ``model.bin``) and the original source to a
   target Hugging Face repo, with a per-model model card.

genai models (``src: genai`` in ``manifests/genai.yaml``) are whole
openvino-genai directories rather than a single IR: they are snapshotted from
their ``upstream`` repo and copied under ``genai/<name>/`` in the mirror, so
ovkit serves them from your mirror too (with the upstream repo as a fallback).

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
import time
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


def _license_text(entry: ModelEntry) -> str:
    """Return the upstream LICENSE/attribution text to ship beside the model.

    Permissive redistribution requires preserving the license and attribution,
    so we fetch the original license text where we can, with a recorded
    attribution fallback.
    """
    if entry.license_url:
        try:
            return _http_text(entry.license_url)
        except Exception:
            pass
    if entry.src == "hf" and entry.repo:
        try:
            from huggingface_hub import hf_hub_download

            for cand in ("LICENSE", "LICENSE.txt", "NOTICE"):
                try:
                    lp = hf_hub_download(repo_id=entry.repo, filename=cand)
                    return Path(lp).read_text(encoding="utf-8", errors="replace")
                except Exception:
                    continue
        except Exception:
            pass
    return (
        f"{entry.name}\nLicense: {entry.license}\n"
        f"Source: {entry.repo or entry.url}\nLicense URL: {entry.license_url or 'n/a'}\n"
    )


def _genai_base(entry: ModelEntry) -> str:
    return entry.subfolder or f"genai/{entry.name}"


def _genai_manifest_snippet(entry: ModelEntry, repo: str) -> str:
    """Emit a genai.yaml entry: mirror as primary, upstream repo as fallback."""
    upstream = entry.extra.get("upstream") or entry.repo
    lines = [
        f"{entry.name}:",
        "  src: genai",
        f"  pipeline: {entry.extra.get('pipeline')}",
        f"  repo: {repo}",
        f"  subfolder: {_genai_base(entry)}",
        f"  license: {entry.license}",
    ]
    if upstream and upstream != repo:
        lines.append(f"  upstream: {upstream}")
    return "\n".join(lines)


def _genai_card(entry: ModelEntry) -> str:
    upstream = entry.extra.get("upstream") or entry.repo
    return textwrap.dedent(f"""\
        ---
        license: {entry.license}
        tags:
          - openvino
          - ovkit
          - genai
          - {entry.extra.get('pipeline')}
        ---

        # {entry.name}

        openvino-genai `{entry.extra.get('pipeline')}` pipeline mirrored for
        [ovkit](https://github.com/leeyunjai82/ovkit) from `{upstream}`.

        ```python
        from ovkit.genai import pipeline
        p = pipeline("{entry.name}")
        ```
        """)


def _genai_operations_for(entry: ModelEntry, repo: str) -> tuple[list, str]:
    """Snapshot a genai model from its upstream repo and stage it under the mirror.

    genai models are whole directories (no IR conversion); we copy every file
    into ``<subfolder>/`` in the mirror so ovkit downloads them from there.
    """
    from huggingface_hub import CommitOperationAdd, snapshot_download

    upstream = entry.extra.get("upstream") or entry.repo
    base = _genai_base(entry)
    print(f"\n=== {entry.name} (genai/{entry.extra.get('pipeline')}, {entry.license}) ===")
    print(f"  snapshot {upstream} ...")
    local = Path(snapshot_download(repo_id=upstream))
    ops = []
    for f in sorted(local.rglob("*")):
        if f.is_file():
            rel = f.relative_to(local).as_posix()
            ops.append(CommitOperationAdd(path_in_repo=f"{base}/{rel}", path_or_fileobj=str(f)))
    ops.append(
        CommitOperationAdd(
            path_in_repo=f"{base}/README.md", path_or_fileobj=_genai_card(entry).encode("utf-8")
        )
    )
    print(f"  staged {len(ops)} files -> {base}/")
    return ops, _genai_manifest_snippet(entry, repo)


def _operations_for(entry: ModelEntry, repo: str, precision: str) -> tuple[list, str]:
    """Download + convert one model and return its commit operations + snippet.

    Returns ``(operations, manifest_snippet)`` where ``operations`` is a list of
    ``CommitOperationAdd`` referencing cached files (no copies). All operations
    are committed together in batches so we make a handful of API calls instead
    of one per file (which trips HF's rate limit).
    """
    if entry.src == "genai":
        return _genai_operations_for(entry, repo)

    from huggingface_hub import CommitOperationAdd

    print(f"\n=== {entry.name} ({entry.task}, {entry.license}) ===")
    source = fetch(entry)
    ir_xml = to_ir(source, entry.name, precision)
    ir_bin = ir_xml.with_suffix(".bin")
    print(f"  source : {source}\n  IR     : {ir_xml}")

    xml_path, bin_path, base = _mirror_paths(entry)
    ops = [CommitOperationAdd(path_in_repo=xml_path, path_or_fileobj=str(ir_xml))]
    if ir_bin.is_file():
        ops.append(CommitOperationAdd(path_in_repo=bin_path, path_or_fileobj=str(ir_bin)))
    ops.append(
        CommitOperationAdd(path_in_repo=f"{base}/{source.name}", path_or_fileobj=str(source))
    )
    ops.append(
        CommitOperationAdd(
            path_in_repo=f"{base}/README.md",
            path_or_fileobj=_model_card(entry, source.name).encode("utf-8"),
        )
    )
    ops.append(
        CommitOperationAdd(
            path_in_repo=f"{base}/LICENSE",
            path_or_fileobj=_license_text(entry).encode("utf-8"),
        )
    )
    print(f"  staged {len(ops)} files -> {base}/")
    return ops, _manifest_snippet(entry, repo)


def _dedupe_ops(operations: list) -> list:
    """Keep one operation per repo path (last wins) to avoid dup-in-commit."""
    by_path: dict[str, object] = {}
    for op in operations:
        by_path[op.path_in_repo] = op
    return list(by_path.values())


def _commit_in_batches(api, repo: str, operations: list, batch: int = 200) -> None:
    """Commit operations in a few large batches, backing off on HF 429s.

    HF rate-limits *commits*, so we make as few as possible (large batches) and
    back off patiently. If the account is in a rate-limit cooldown (e.g. from an
    earlier many-commit run), even this can fail — re-running later resumes
    cheaply because uploaded blobs are cached.
    """
    operations = _dedupe_ops(operations)
    total = len(operations)
    for start in range(0, total, batch):
        chunk = operations[start : start + batch]
        n = start // batch + 1
        for attempt in range(8):
            try:
                api.create_commit(
                    repo_id=repo,
                    repo_type="model",
                    operations=chunk,
                    commit_message=f"ovkit mirror upload (batch {n})",
                )
                print(f"  committed batch {n}: {len(chunk)} files ({start + len(chunk)}/{total})")
                break
            except Exception as exc:  # backoff only on rate limiting
                msg = str(exc).lower()
                if "429" in msg or "rate limit" in msg or "too many requests" in msg:
                    wait = min(120, 10 * 2**attempt)
                    print(f"  rate limited; waiting {wait}s then retrying batch {n}...")
                    time.sleep(wait)
                    continue
                raise
        else:
            raise RuntimeError(
                f"batch {n} still rate-limited after retries. Your HF account is likely "
                f"in a commit rate-limit cooldown (try again in ~30-60 min). Uploaded files "
                f"are cached, so re-running resumes quickly."
            )
        time.sleep(2)  # be gentle between commits


def _repo_readme_text() -> str:
    """Top-level README describing the mirror and its license policy."""
    return textwrap.dedent("""\
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
        attribution are preserved in each model's `LICENSE` file. No AGPL-licensed
        or non-commercial (InsightFace pretrained) weights are hosted here.
        If you believe a model is mis-licensed, please open an issue.
        """)


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
    parser.add_argument(
        "--emit-manifest",
        metavar="PATH",
        help="write a runtime manifest (all selected models, mirror primary + "
        "upstream fallback) to PATH and exit; no download/upload",
    )
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

    # Emit a ready-to-commit runtime manifest without downloading/uploading.
    if args.emit_manifest:
        header = (
            "# Generated by scripts/build_mirror.py --emit-manifest\n"
            f"# Mirror: {args.repo} (primary) with upstream fallback.\n\n"
        )
        # genai models are configured in manifests/genai.yaml, not here; emitting
        # them as IR entries would clobber that. Keep this manifest IR-only.
        ir_entries = [e for e in entries if e.src != "genai"]
        body = "\n\n".join(_manifest_snippet(e, args.repo) for e in ir_entries)
        Path(args.emit_manifest).write_text(header + body + "\n", encoding="utf-8")
        print(f"Wrote {len(ir_entries)} entries to {args.emit_manifest}")
        return 0

    api = None
    if not args.dry_run:
        try:
            from huggingface_hub import HfApi
        except ImportError:
            print("huggingface_hub is required (pip install ovkit).", file=sys.stderr)
            return 1
        api = HfApi(token=os.environ.get("HF_TOKEN"))
        api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)

    # Download + convert each model and collect its commit operations.
    from huggingface_hub import CommitOperationAdd

    all_ops: list = [
        CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=_repo_readme_text().encode())
    ]
    snippets, failed = [], []
    for entry in entries:
        try:
            ops, snippet = _operations_for(entry, args.repo, args.precision)
            all_ops.extend(ops)
            snippets.append(snippet)
        except Exception as exc:  # keep going across the remaining models
            failed.append(entry.name)
            print(f"  ERROR staging '{entry.name}': {exc}", file=sys.stderr)

    if not args.dry_run:
        print(f"\nUploading {len(all_ops)} files in batches (rate-limit safe)...")
        _commit_in_batches(api, args.repo, all_ops)

    if snippets:
        print("\n" + "=" * 70)
        print("Manifest entries pointing at the mirror (paste into models.yaml):\n")
        print("\n\n".join(snippets))
    if args.dry_run:
        print(
            f"\nDRY-RUN: {len(snippets)} checked (downloaded + converted, "
            f"NOT uploaded), {len(failed)} failed."
        )
        print("Re-run WITHOUT --dry-run (and with a valid HF_TOKEN) to upload.")
    else:
        print(f"\nDone: {len(snippets)} models uploaded to {args.repo}, {len(failed)} failed.")
    if failed:
        print("Failed: " + ", ".join(failed), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
