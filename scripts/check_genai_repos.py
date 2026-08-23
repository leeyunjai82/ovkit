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


def _org_search(org: str, term: str, limit: int = 5) -> list[str]:
    """Model ids in ``org`` matching ``term`` (Hugging Face search API)."""
    q = urllib.parse.urlencode({"author": org, "search": term, "limit": limit})
    data = _get(SEARCH + q)
    return [m["id"] for m in data] if isinstance(data, list) else []


def _suggest(name: str, limit: int = 6) -> list[str]:
    """Close matches in the same org, to fix a wrong id without guessing.

    Tries progressively broader terms taken from the id — the longest prefixes
    first ("Qwen2-VL-2B" before "Qwen2") — because searching only the first
    token surfaces unrelated families (Qwen2.5 LLMs for a Qwen2-VL request).
    """
    org, _, model = name.partition("/")
    parts = [p for p in model.split("-") if p and p.lower() not in {"ov", "int4", "int8", "fp16"}]
    terms: list[str] = []
    for n in range(min(3, len(parts)), 0, -1):  # "A-B-C", "A-B", "A"
        terms.append("-".join(parts[:n]))
    terms += [p for p in parts[1:] if len(p) > 2]  # distinctive middle tokens ("VL", "tts")

    seen, out = set(), []
    for term in terms:
        for mid in _org_search(org, term, limit):
            if mid not in seen:
                seen.add(mid)
                out.append(mid)
        if len(out) >= limit:
            break
    return out[:limit]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Check genai upstream repos on Hugging Face.")
    ap.add_argument("--prune", action="store_true", help="remove entries whose upstream is missing")
    ap.add_argument(
        "--search",
        metavar="TERM",
        help="just search the OpenVINO org for TERM and exit (find a replacement id)",
    )
    ap.add_argument("--org", default="OpenVINO", help="org to search (default: OpenVINO)")
    args = ap.parse_args(argv)

    if args.search:
        hits = _org_search(args.org, args.search, limit=25)
        if not hits:
            print(f"No models matching {args.search!r} in the {args.org} org.")
            return 1
        print(f"{len(hits)} match(es) for {args.search!r} in {args.org}:")
        for mid in hits:
            info = _get(API + mid)
            lic = ((info or {}).get("cardData") or {}).get("license", "?")
            print(f"  {mid:60s} {lic}")
        return 0

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
