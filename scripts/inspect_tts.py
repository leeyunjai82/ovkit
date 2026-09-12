#!/usr/bin/env python3
"""Look inside a TTS candidate before trying to convert it.

    python scripts/inspect_tts.py                       # the chosen candidate
    python scripts/inspect_tts.py owner/model ...

Prints what the repository actually contains — file names and sizes, the
config's architecture, and what a conversion would have to deal with. Nothing
is downloaded beyond the metadata and small text files.

Why look first
--------------
Today's Korean OCR work started with a confident guess about a model's class
numbering and ended with "building" decoded as "c0v0j0me0joh0". The cost of
guessing is a day of work thrown away at the last step, and the cure is thirty
seconds of looking. So before writing a converter for MeloTTS: does it ship
ONNX already, or only PyTorch? Does the text front end need extra packages a
classroom machine would not have?

Runs in Actions (**Check candidates** workflow) — the development container
cannot reach huggingface.co.
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request

API = "https://huggingface.co/api/models/"
RAW = "https://huggingface.co/{model}/resolve/main/{path}"

DEFAULT = ["myshell-ai/MeloTTS-Korean"]

#: Files worth reading in full — they say what the model expects, and under
#: what terms. LICENSE is first because a licence decides whether the rest
#: matters at all.
SMALL_TEXT = (
    "LICENSE",
    "config.json",
    "config.yml",
    "config.yaml",
    "tokenizer_config.json",
)


def _get_json(url: str) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        print(f"  !! HTTP {exc.code} for {url}")
    except Exception as exc:  # noqa: BLE001
        print(f"  !! {type(exc).__name__} for {url}")
    return None


def _human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:,.0f} {unit}" if unit == "B" else f"{size:,.1f} {unit}"
        size /= 1024
    return f"{size:,.1f} GB"


def inspect(model_id: str) -> None:
    print(f"\n=== {model_id} " + "=" * max(0, 56 - len(model_id)))
    info = _get_json(API + model_id)
    if info is None:
        return

    card = info.get("cardData") or {}
    print(f"  licence   : {card.get('license', '?')}")
    print(f"  pipeline  : {info.get('pipeline_tag', '?')}")
    print(f"  library   : {info.get('library_name', '?')}")
    print(f"  downloads : {info.get('downloads', 0):,}")

    files = [f for f in (info.get("siblings") or []) if isinstance(f, dict)]
    print(f"  {len(files)} file(s):")
    for entry in sorted(files, key=lambda f: str(f.get("rfilename"))):
        name = str(entry.get("rfilename"))
        size = entry.get("size")
        print(f"    {(_human(size) if size else '?'):>10}  {name}")

    names = {str(f.get("rfilename")) for f in files}
    onnx = sorted(n for n in names if n.endswith(".onnx"))
    torch = sorted(n for n in names if n.endswith((".pth", ".bin", ".safetensors", ".ckpt")))
    print(f"\n  ONNX      : {', '.join(onnx) if onnx else '없음 — 직접 export 해야 합니다'}")
    print(f"  PyTorch   : {', '.join(torch) if torch else '없음'}")

    for path in SMALL_TEXT:
        if path not in names:
            continue
        try:
            with urllib.request.urlopen(
                RAW.format(model=model_id, path=path), timeout=30
            ) as r:  # noqa: S310
                body = r.read().decode("utf-8", "replace")
        except Exception as exc:  # noqa: BLE001
            print(f"  {path}: 못 읽음 ({type(exc).__name__})")
            continue
        print(f"\n  --- {path} (앞부분) ---")
        print("\n".join("  " + line for line in body.splitlines()[:40]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("ids", nargs="*", default=DEFAULT, help="model ids")
    args = parser.parse_args()
    for model_id in args.ids or DEFAULT:
        inspect(model_id)
    print(
        "\n변환 전에 확인할 것: ONNX가 이미 있는지, 없으면 어떤 그래프를 export해야 하는지,"
        "\n그리고 텍스트 프론트엔드(g2p·자모 분해)가 교실 컴퓨터에 없는 패키지를 요구하는지."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
