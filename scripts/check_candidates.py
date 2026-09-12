#!/usr/bin/env python3
"""Ask Hugging Face what a candidate model actually is, before committing to it.

    python scripts/check_candidates.py                    # the standing shortlist
    python scripts/check_candidates.py owner/model ...    # anything else

Prints each model's licence, size and popularity, and says whether ovkit's
policy would let it load at all — ``PERMISSIVE_LICENSES`` in
``core/constants.py`` refuses AGPL and non-commercial weights, and a model
that fails there is not a candidate however good it is.

Why a script instead of remembering
-----------------------------------
Licences are the thing that is easiest to be confidently wrong about: MMS-TTS
reads as "Meta, open" and is CC-BY-NC-4.0; several Korean TTS checkpoints are
research-only. Guessing produces work that has to be thrown away at the last
step, so the shortlist below is checked against the Hub rather than against
anyone's memory.

The development container cannot reach huggingface.co, so this is written to
run in Actions (the **Check candidates** workflow) — the same place the
conversions run.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request

API = "https://huggingface.co/api/models/"

#: What ovkit is shopping for, and why. Nothing here is chosen yet.
SHORTLIST: dict[str, list[str]] = {
    "한국어 TTS (읽어주기)": [
        "myshell-ai/MeloTTS-Korean",
        "facebook/mms-tts-kor",
        "espnet/kan-bayashi_ksponspeech_vits",
        "Supertone/supertonic",
        "microsoft/speecht5_tts",
    ],
    "한국어 감정·문장 분류": [
        "monologg/koelectra-small-v3-discriminator",
        "beomi/KcELECTRA-base",
        "klue/roberta-small",
        "matthewburke/korean_sentiment",
    ],
    "한국어 개체명": [
        "KPF/KPF-bert-ner",
        "Leo97/KoELECTRA-small-v3-modu-ner",
    ],
    "한국어 QA": [
        "monologg/koelectra-small-v2-distilled-korquad-384",
        "bespin-global/klue-bert-base-aihub-mrc",
    ],
    "한국어 문장 임베딩": [
        "jhgan/ko-sroberta-multitask",
        "BM-K/KoSimCSE-roberta",
    ],
}


def ask(model_id: str) -> dict | None:
    try:
        with urllib.request.urlopen(API + model_id, timeout=30) as response:  # noqa: S310
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return {"error": f"HTTP {exc.code}"}
    except Exception as exc:  # noqa: BLE001
        return {"error": type(exc).__name__}


def licence_of(info: dict) -> str:
    for tag in info.get("tags", []):
        if isinstance(tag, str) and tag.startswith("license:"):
            return tag.split(":", 1)[1]
    card = info.get("cardData") or {}
    value = card.get("license")
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value) if value else "?"


def size_of(info: dict) -> str:
    total = sum(
        int(f.get("size") or 0) for f in (info.get("siblings") or []) if isinstance(f, dict)
    )
    if not total:
        used = info.get("usedStorage")
        total = int(used) if used else 0
    if not total:
        return "?"
    for unit, step in (("GB", 1024**3), ("MB", 1024**2), ("KB", 1024)):
        if total >= step:
            return f"{total / step:.0f} {unit}"
    return f"{total} B"


def _is_permissive():
    """ovkit's licence policy, loaded without importing ovkit.

    ``import ovkit.core.constants`` pulls the package in, and the package
    pulls in numpy and OpenVINO — a licence lookup should not need either.
    The module itself is pure stdlib, so it is loaded straight from the file.
    """
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "src" / "ovkit" / "core" / "constants.py"
    spec = importlib.util.spec_from_file_location("ovkit_constants", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["ovkit_constants"] = module
    spec.loader.exec_module(module)
    return module.is_permissive


def report(groups: dict[str, list[str]]) -> int:
    is_permissive = _is_permissive()

    blocked = 0
    for heading, ids in groups.items():
        print(f"\n{heading}")
        print("-" * len(heading.encode("utf-8")))
        for model_id in ids:
            info = ask(model_id)
            if info is None or "error" in (info or {}):
                print(f"  {'없음':6s} {model_id}   ({(info or {}).get('error', 'unreachable')})")
                continue
            licence = licence_of(info)
            ok = is_permissive(licence)
            blocked += 0 if ok else 1
            # "미표기" and "거부" both end up blocked, but they are not the same
            # finding: one is a licence ovkit will not accept, the other is a
            # model card that never said. Reporting them as one hides which
            # ones a maintainer could resolve by asking.
            if ok:
                mark, licence = "OK", licence
            elif licence in {"?", "None", ""}:
                mark, licence = "미표기", "(모델 카드에 없음)"
            else:
                mark = "거부"
            downloads = info.get("downloads") or 0
            print(f"  {mark:6s} {model_id:52s} {licence:22s} {size_of(info):>8s}  ↓{downloads:,}")
    print(
        "\n'거부'  ovkit 정책이 받지 않는 라이선스입니다 — 성능과 무관하게 후보가 아닙니다."
        "\n'미표기' 모델 카드에 라이선스가 없습니다. 없는 것을 허용으로 가정할 수는 없으니"
        "\n        정책은 똑같이 막지만, 이쪽은 올린 사람에게 물어보면 풀릴 수도 있습니다."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("ids", nargs="*", help="model ids to check (default: the shortlist)")
    args = parser.parse_args()
    groups = {"직접 지정": args.ids} if args.ids else SHORTLIST
    return report(groups)


if __name__ == "__main__":
    raise SystemExit(main())
