#!/usr/bin/env python3
"""Read Supertonic's four graphs and its JSON tables, and say what they expect.

    python scripts/inspect_supertonic.py

Downloads the repository (small — four ONNX graphs and some JSON), prints every
graph's inputs and outputs with shapes and dtypes, and summarises the tables
that turn text into ids. Converts each graph to OpenVINO IR to prove it can be.

Why this comes before any pipeline code
---------------------------------------
Twice today a confident guess about a model's interface cost a rewrite: the
Korean OCR alphabet (blank first or last) and the TTS export signature. A TTS
pipeline chains four graphs; guessing four interfaces would be four times the
mistake. So the interfaces get read, printed, and pasted into the
implementation — not remembered.

Runs in Actions (**Supertonic** workflow); this container cannot reach
huggingface.co.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

#: Default repository. ``Supertone/supertonic-3`` is the newer release and
#: the one the PyPI example loads; the older ``Supertone/supertonic`` ships
#: the ``opensource-en`` split, whose character table is ASCII-only.
REPO = "Supertone/supertonic-3"
GRAPHS = (
    "onnx/text_encoder.onnx",
    "onnx/duration_predictor.onnx",
    "onnx/vector_estimator.onnx",
    "onnx/vocoder.onnx",
)
TABLES = ("onnx/tts.json", "onnx/tts.yml", "onnx/unicode_indexer.json", "voice_styles/F1.json")


def _human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:,.0f} {unit}" if unit == "B" else f"{size:,.1f} {unit}"
        size /= 1024
    return f"{size:,.1f} GB"


def _peek(value: object, depth: int = 0) -> str:
    """A short shape-of-the-data summary, not a dump."""
    pad = "  " * depth
    if isinstance(value, dict):
        keys = list(value)
        head = ", ".join(map(str, keys[:8]))
        more = f" … (+{len(keys) - 8})" if len(keys) > 8 else ""
        lines = [f"{pad}dict[{len(keys)}]: {head}{more}"]
        for key in keys[:3]:
            lines.append(f"{pad}  {key} -> {_peek(value[key], depth + 1).strip()}")
        return "\n".join(lines)
    if isinstance(value, list):
        # A voice style is a few thousand floats. Printing six of them says
        # nothing and printing all of them buried the graph interfaces under
        # 300 KB of numbers.
        if value and all(isinstance(v, (int, float)) for v in value):
            return f"{pad}list[{len(value)}] of numbers"
        if value and all(isinstance(v, list) for v in value):
            inner = len(value[0])
            return f"{pad}list[{len(value)}] x list[{inner}] of numbers"
        sample = ", ".join(repr(v)[:40] for v in value[:6])
        return f"{pad}list[{len(value)}]: {sample}{' …' if len(value) > 6 else ''}"
    return f"{pad}{type(value).__name__}: {str(value)[:80]}"



#: Unicode blocks worth asking about by name. ``tts.json`` says the released
#: split is ``opensource-en``; whether that means "English voices" or "English
#: characters only" is the difference between a Korean TTS and a dead end, and
#: the indexer answers it without a guess.
_BLOCKS = {
    "ASCII 소문자 a-z": (0x61, 0x7A),
    "숫자 0-9": (0x30, 0x39),
    "한글 자모 ㄱ-ㅎ": (0x3131, 0x314E),
    "한글 음절 가-힣": (0xAC00, 0xD7A3),
    "한자": (0x4E00, 0x9FFF),
    "일본어 히라가나": (0x3041, 0x3096),
}


def _can_it_say_hangul() -> None:
    """Ask the unicode indexer which characters it actually has ids for.

    A TTS that cannot index 가-힣 cannot read Korean, whatever its voices sound
    like. Cheaper to learn here than after a pipeline is written around it.
    """
    from huggingface_hub import hf_hub_download

    print("--- 이 모델이 읽을 수 있는 글자 (unicode_indexer.json)")
    try:
        table = json.loads(
            Path(hf_hub_download(REPO, "onnx/unicode_indexer.json")).read_text(encoding="utf-8")
        )
    except Exception as exc:  # noqa: BLE001
        print(f"    !! {type(exc).__name__}: {str(exc)[:200]}")
        return
    if not isinstance(table, list):
        print(f"    예상과 다른 모양: {type(table).__name__}")
        return
    known = [i for i, v in enumerate(table) if isinstance(v, int) and v >= 0]
    print(f"    표 길이 {len(table):,} / 실제로 id가 붙은 글자 {len(known):,}개")
    print(f"    id 범위 0..{max((table[i] for i in known), default=-1)}")
    for label, (lo, hi) in _BLOCKS.items():
        got = sum(1 for cp in range(lo, hi + 1) if cp < len(table) and table[cp] >= 0)
        total = hi - lo + 1
        mark = "O" if got == total else ("일부" if got else "X")
        print(f"    {label:16s} {got:5d}/{total:<5d}  {mark}")
    sample = "".join(chr(i) for i in known[:80] if 0x20 <= i < 0x3000)
    print(f"    id가 붙은 글자 맛보기: {sample[:80]!r}")
    print()



def _how_is_it_driven() -> None:
    """Print the repository's own files and README.

    The four interfaces say what each graph eats. They do not say how many
    flow-matching steps to run, how ``duration`` becomes a latent length, or
    whether ``denoised_latent`` is the vector field or the updated latent. That
    is the repository's to answer, and it answers in prose and file names.
    """
    from huggingface_hub import hf_hub_download, list_repo_files

    print("--- 저장소에 실제로 들어 있는 파일")
    try:
        files = sorted(list_repo_files(REPO))
    except Exception as exc:  # noqa: BLE001
        print(f"    !! {type(exc).__name__}: {str(exc)[:200]}")
        files = []
    for name in files:
        print(f"    {name}")
    print()

    for doc in ("README.md",):
        if files and doc not in files:
            print(f"--- {doc}: 없음")
            continue
        try:
            text = Path(hf_hub_download(REPO, doc)).read_text(encoding="utf-8", errors="replace")
        except Exception as exc:  # noqa: BLE001
            print(f"--- {doc}: {type(exc).__name__}")
            continue
        print(f"--- {doc} ({len(text):,}자)")
        print("\n".join("  " + line for line in text.splitlines()[:200]))
        print()


def main() -> int:
    global REPO

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=REPO, help="Hugging Face 저장소 id")
    REPO = parser.parse_args().repo

    from huggingface_hub import hf_hub_download

    print(f"=== {REPO} ===\n")

    for name in GRAPHS:
        path = Path(hf_hub_download(REPO, name))
        print(f"--- {name}  ({_human(path.stat().st_size)})")
        try:
            import openvino as ov

            model = ov.convert_model(str(path))
            for port in model.inputs:
                print(
                    f"    in   {port.any_name:24s} {port.element_type.get_type_name():8s} "
                    f"{port.partial_shape}"
                )
            for port in model.outputs:
                print(
                    f"    out  {port.any_name:24s} {port.element_type.get_type_name():8s} "
                    f"{port.partial_shape}"
                )
            print("    -> OpenVINO 변환 OK")
        except Exception as exc:  # noqa: BLE001 - the failure is the finding
            print(f"    !! {type(exc).__name__}: {str(exc)[:200]}")
        print()

    for name in TABLES:
        try:
            path = Path(hf_hub_download(REPO, name))
        except Exception as exc:  # noqa: BLE001
            print(f"--- {name}: 없음 ({type(exc).__name__})")
            continue
        print(f"--- {name}  ({_human(path.stat().st_size)})")
        text = path.read_text(encoding="utf-8", errors="replace")
        if name.endswith(".json"):
            try:
                print(_peek(json.loads(text)))
            except Exception:  # noqa: BLE001
                print(text[:400])
        else:
            # Only the lines that name a component or a file: the rest is
            # training configuration, and dumping it buried the interfaces in
            # 300 KB of log.
            wanted = [
                line
                for line in text.splitlines()
                if line.strip().endswith(":") or "path" in line or "onnx" in line
            ]
            print("\n".join("  " + line for line in wanted[:40]))
        print()

    _can_it_say_hangul()
    _how_is_it_driven()

    print(
        "이 표에 적힌 입출력 이름과 모양이 파이프라인 구현의 근거다."
        "\n기억이 아니라 여기서 복사해 쓴다."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
