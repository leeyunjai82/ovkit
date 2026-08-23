#!/usr/bin/env python3
"""Verify the genai manifest against Hugging Face — no downloads, seconds.

Each ``src: genai`` entry names an ``upstream`` repo on the OpenVINO org. This
checks that every one exists and that the licence ovkit declares matches the
licence Hugging Face reports (ovkit only serves permissive models, so a
mismatch matters). Missing repos get close-match suggestions from the same
org, so a wrong id is a one-line fix instead of a guessing game::

    python scripts/check_genai_repos.py            # report
    python scripts/check_genai_repos.py --prune    # drop entries that 404

Run it before ``build_mirror.py`` — mirroring a genai model downloads GBs, so
it is worth spending five seconds first.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import yaml

MANIFEST = Path(__file__).resolve().parent.parent / "src" / "ovkit" / "manifests" / "genai.yaml"
API = "https://huggingface.co/api/models/"
SEARCH = "https://huggingface.co/api/models?"
OK, NO, WARN = "✅", "❌", "⚠️ "


def _get(url: str) -> object | None:
    req = urllib.request.Request(url, headers={"User-Agent": "ovkit-check"})  # noqa: S310
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:  # noqa: S310
            return json.load(resp)
    except Exception:
        return None


def _suggest(name: str, limit: int = 5) -> list[str]:
    """Close matches in the same org, to fix a wrong id without guessing."""
    org, _, model = name.partition("/")
    stem = model.split("-")[0] or model
    q = urllib.parse.urlencode({"author": org, "search": stem, "limit": limit})
    data = _get(SEARCH + q)
    return [m["id"] for m in data] if isinstance(data, list) else []


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Check genai upstream repos on Hugging Face.")
    ap.add_argument("--prune", action="store_true", help="remove entries whose upstream is missing")
    args = ap.parse_args(argv)

    if _get(SEARCH + "limit=1") is None:
        print(
            "Cannot reach huggingface.co — every model would look 'missing', so this\n"
            "check would be meaningless (and --prune would delete good entries).\n"
            "Run it from a machine/network that can reach Hugging Face.",
            file=sys.stderr,
        )
        return 2

    manifest = yaml.safe_load(MANIFEST.read_text(encoding="utf-8")) or {}
    genai = {k: v for k, v in manifest.items() if isinstance(v, dict) and v.get("src") == "genai"}
    if not genai:
        print("No genai entries found.")
        return 0

    missing: list[str] = []
    print(f"Checking {len(genai)} genai models against Hugging Face...\n")
    for name, spec in genai.items():
        upstream = spec.get("upstream") or spec.get("repo") or ""
        info = _get(API + upstream)
        if info is None:
            missing.append(name)
            print(f"{NO} {name:32s} {upstream}  — not found")
            for s in _suggest(upstream):
                print(f"     ↳ maybe: {s}")
            continue
        hf_license = (
            ((info.get("cardData") or {}).get("license") or "?") if isinstance(info, dict) else "?"
        )
        declared = str(spec.get("license", "?")).lower()
        mark = OK if str(hf_license).lower() == declared else WARN
        note = "" if mark == OK else f"  (manifest says {declared}, HF says {hf_license})"
        print(f"{mark} {name:32s} {upstream}{note}")

    print()
    if missing:
        print(f"{len(missing)} missing: {', '.join(missing)}")
        if args.prune:
            text = MANIFEST.read_text(encoding="utf-8")
            for name in missing:
                # drop the whole block: "name:" through the blank line before the next entry
                lines, out, skip = text.splitlines(keepends=True), [], False
                for ln in lines:
                    if ln.startswith(f"{name}:"):
                        skip = True
                        continue
                    if skip and ln and not ln[0].isspace() and not ln.startswith("#"):
                        skip = False
                    if not skip:
                        out.append(ln)
                text = "".join(out)
            MANIFEST.write_text(text, encoding="utf-8")
            print(f"Pruned {len(missing)} entr(ies) from {MANIFEST.name}.")
        else:
            print("Fix the ids above (or re-run with --prune to drop them).")
        return 1
    print("All genai upstream repos exist.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
