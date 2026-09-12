#!/usr/bin/env python3
"""Find a Korean text-to-speech model ovkit is actually allowed to ship.

    python scripts/find_korean_tts.py

Asks the Hugging Face Hub for text-to-speech models tagged Korean, and for each
one prints the three things that decide whether ovkit can use it:

* **licence** — ovkit serves permissive licences, plus OpenRAIL-M when the
  obligation can travel with the model. CC-BY-NC and AGPL are out, however good
  the voice is: ovkit goes into schools and a model ovkit cannot pass on is a
  model ovkit will not load.
* **what it ships** — ONNX or a PyTorch checkpoint. ONNX converts to OpenVINO
  IR in one call; a PyTorch checkpoint needs its training code, its tokenizer
  and usually a Korean g2p package, which is a dependency ovkit would be asking
  every student to install.
* **whether anyone uses it** — downloads, as a rough proxy for "this works".

Why this script exists: Supertonic turned out to be English-only (its character
table has ids for 81 ASCII characters and none for 가-힣), so the Korean half of
the job needs a different model. Picking one from memory is how the OCR
alphabet went wrong; this asks the Hub instead.

Runs in Actions (**Korean TTS** workflow); this container cannot reach
huggingface.co.
"""

from __future__ import annotations

import argparse
import re
from typing import Any

#: Licences ovkit can serve, mirrored from ``ovkit.core.constants``. Kept as a
#: literal so this script runs in a bare Actions job with no ovkit installed.
OK_LICENSES = {
    "apache-2.0",
    "mit",
    "bsd",
    "bsd-2-clause",
    "bsd-3-clause",
    "isc",
    "unlicense",
    "cc0-1.0",
    "mpl-2.0",
    "openrail",
    "openrail-m",
    "bigscience-openrail-m",
    "cc-by-4.0",
}

#: Searches to run. The Hub's own tags are inconsistent for TTS, so ask several
#: ways rather than trusting one filter to be complete.
QUERIES: tuple[tuple[str, dict[str, Any]], ...] = (
    ("tts + ko 태그", {"filter": ["text-to-speech", "ko"]}),
    ("tts 태그 + korean 검색", {"filter": ["text-to-speech"], "search": "korean"}),
    ("tts 태그 + ko 검색", {"filter": ["text-to-speech"], "search": "ko"}),
    ("vits + korean", {"search": "vits korean"}),
    ("piper + ko", {"search": "piper ko"}),
    ("melotts", {"search": "melotts"}),
)



#: A filename counts as Korean only on a word boundary. The first version of
#: this check matched a bare "ko" and duly reported ``kokoro-v1_0.pth`` and
#: ``voices/af_kore.pt`` as Korean files, which is the kind of confident wrong
#: answer this whole script exists to avoid.
_KOREAN = re.compile(r"(?:^|[/_.\-])(?:ko|kor|korean|ko[_-]?kr)(?:$|[/_.\-])", re.IGNORECASE)


def _looks_korean(filename: str) -> bool:
    return bool(_KOREAN.search(filename))


def _license_of(info: Any) -> str:
    for tag in getattr(info, "tags", None) or []:
        if isinstance(tag, str) and tag.startswith("license:"):
            return tag.split(":", 1)[1]
    card = getattr(info, "cardData", None) or {}
    return str(card.get("license") or "?")


def _verdict(license_id: str) -> str:
    if license_id.lower() in OK_LICENSES:
        return "쓸 수 있음"
    if license_id in {"?", ""}:
        return "라이선스 불명"
    return "못 씀"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=25, help="검색당 최대 개수")
    parser.add_argument(
        "--files", action="store_true", help="쓸 수 있는 후보의 파일 목록도 본다"
    )
    args = parser.parse_args()

    from huggingface_hub import HfApi

    api = HfApi()
    seen: dict[str, tuple[str, int]] = {}

    for label, kwargs in QUERIES:
        print(f"=== {label}")
        try:
            models = list(api.list_models(limit=args.limit, sort="downloads", **kwargs))
        except Exception as exc:  # noqa: BLE001 - the failure is the finding
            print(f"    !! {type(exc).__name__}: {str(exc)[:200]}\n")
            continue
        if not models:
            print("    (없음)\n")
            continue
        for info in models:
            license_id = _license_of(info)
            downloads = int(getattr(info, "downloads", 0) or 0)
            verdict = _verdict(license_id)
            print(f"    {verdict:12s} {license_id:22s} {downloads:>9,}  {info.id}")
            if verdict == "쓸 수 있음":
                seen[info.id] = (license_id, downloads)
        print()

    print(f"=== 쓸 수 있는 후보 {len(seen)}개 (내려받은 수 순)")
    ranked = sorted(seen.items(), key=lambda kv: -kv[1][1])
    for name, (license_id, downloads) in ranked:
        print(f"    {license_id:22s} {downloads:>9,}  {name}")

    if args.files:
        print("\n=== 후보들이 실제로 담고 있는 것")
        for name, _ in ranked[:12]:
            try:
                files = sorted(api.list_repo_files(name))
            except Exception as exc:  # noqa: BLE001
                print(f"--- {name}: {type(exc).__name__}")
                continue
            onnx = [f for f in files if f.endswith(".onnx")]
            weights = [f for f in files if f.endswith((".bin", ".safetensors", ".pth", ".pt"))]
            print(f"--- {name}")
            print(f"    ONNX {len(onnx)}개, 파이토치 가중치 {len(weights)}개, 파일 {len(files)}개")
            for f in (onnx + weights)[:8]:
                print(f"      {f}")
            korean = [f for f in files if _looks_korean(f)]
            if korean:
                print(f"    한국어로 보이는 파일: {', '.join(korean[:6])}")

    print(
        "\n고르는 기준: ONNX를 이미 갖고 있으면 OpenVINO로 한 번에 변환된다."
        "\n파이토치만 있으면 학습 코드와 한국어 g2p 패키지까지 따라오는데,"
        "\n그건 학생 노트북마다 설치해야 하는 짐이다."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
