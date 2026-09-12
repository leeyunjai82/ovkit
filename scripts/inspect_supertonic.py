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

import json
from pathlib import Path

REPO = "Supertone/supertonic"
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
        sample = ", ".join(repr(v) for v in value[:6])
        return f"{pad}list[{len(value)}]: {sample}{' …' if len(value) > 6 else ''}"
    return f"{pad}{type(value).__name__}: {str(value)[:80]}"


def main() -> int:
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
            print("\n".join("  " + line for line in text.splitlines()[:40]))
        print()

    print(
        "이 표에 적힌 입출력 이름과 모양이 파이프라인 구현의 근거다."
        "\n기억이 아니라 여기서 복사해 쓴다."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
