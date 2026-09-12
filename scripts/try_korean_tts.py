#!/usr/bin/env python3
"""Find out whether MeloTTS-Korean can become an OpenVINO model at all.

    python scripts/try_korean_tts.py

This answers questions, it does not ship anything. Run it from the **Korean
TTS (feasibility)** workflow, where there is a network and room to install
torch.

What it has to establish, in order — each step's failure is the answer
-----------------------------------------------------------------------
1. Does ``pip install`` of the MeloTTS package work at all on a clean machine?
2. Does the Korean checkpoint load?
3. What does the text front end need, and does it reach outside Python (a
   system dictionary, a compiler, mecab)? This decides whether a classroom
   machine can run it, which decides whether ovkit should serve it.
4. Can the synthesiser be traced to ONNX? VITS has a stochastic duration
   predictor and dynamic lengths, which is exactly what tracing is worst at.

ovkit's promise is that a converted model runs with nothing but ovkit. TTS
looks likely to break that promise — the phonemiser is Python, not a graph —
so the point of this script is to replace "likely" with a log.
"""

from __future__ import annotations

import subprocess
import sys
import traceback
from pathlib import Path

TEXT = "안녕하세요. 오늘 날씨가 좋네요."


def step(number: int, what: str) -> None:
    print(f"\n=== {number}. {what} " + "=" * max(0, 50 - len(what)), flush=True)


def run(command: list[str]) -> int:
    print("$ " + " ".join(command), flush=True)
    return subprocess.run(command).returncode


def main() -> int:
    findings: list[str] = []

    step(1, "melotts 설치")
    code = run([sys.executable, "-m", "pip", "install", "--quiet", "melotts"])
    if code != 0:
        findings.append("pip install melotts 실패 — 이 경로는 여기서 끝납니다.")
        print("\n".join(findings))
        return 1
    findings.append("melotts 설치 OK")

    step(2, "한국어 체크포인트 로드")
    try:
        from melo.api import TTS  # type: ignore[import-not-found]

        tts = TTS(language="KR", device="cpu")
        findings.append("체크포인트 로드 OK")
    except Exception:
        traceback.print_exc()
        findings.append("체크포인트 로드 실패 — 아래 트레이스백이 이유입니다.")
        print("\n".join(findings))
        return 1

    step(3, "텍스트 프론트엔드가 무엇을 요구하는가")
    try:
        # What turns 한글 into phonemes, and what does it drag in?
        from melo.text import cleaned_text_to_sequence, get_bert  # noqa: F401
        from melo.text.korean import g2p, text_normalize  # type: ignore[import-not-found]

        normalised = text_normalize(TEXT)
        phones = g2p(normalised)
        print(f"  입력   : {TEXT}")
        print(f"  정규화 : {normalised}")
        print(f"  음소   : {str(phones)[:200]}")
        findings.append("g2p OK — 다만 이것은 그래프가 아니라 파이썬 코드입니다.")
    except Exception:
        traceback.print_exc()
        findings.append("g2p 실패 — 한국어 프론트엔드가 추가 설치를 요구합니다.")

    step(4, "합성기를 ONNX로 추출할 수 있는가")
    try:
        import torch

        model = tts.model
        model.eval()
        out = Path("melotts_kr.onnx")

        # VITS takes phoneme ids, their lengths, tone/language ids and a
        # speaker id. Shapes are dynamic, which is the hard part.
        length = 32
        x = torch.randint(1, 50, (1, length), dtype=torch.long)
        x_len = torch.tensor([length], dtype=torch.long)
        tone = torch.zeros((1, length), dtype=torch.long)
        language = torch.zeros((1, length), dtype=torch.long)
        speaker = torch.tensor([0], dtype=torch.long)
        bert = torch.zeros((1, 1024, length), dtype=torch.float32)
        ja_bert = torch.zeros((1, 768, length), dtype=torch.float32)

        torch.onnx.export(
            model,
            (x, x_len, speaker, tone, language, bert, ja_bert),
            str(out),
            input_names=["x", "x_lengths", "sid", "tone", "language", "bert", "ja_bert"],
            output_names=["audio"],
            dynamic_axes={
                "x": {1: "phonemes"},
                "tone": {1: "phonemes"},
                "language": {1: "phonemes"},
                "bert": {2: "phonemes"},
                "ja_bert": {2: "phonemes"},
                "audio": {2: "samples"},
            },
            opset_version=17,
        )
        size = out.stat().st_size / 1e6
        findings.append(f"ONNX export OK ({size:.0f} MB)")

        step(5, "OpenVINO가 그 ONNX를 읽는가")
        import openvino as ov

        ir = ov.convert_model(str(out))
        for port in ir.inputs:
            print(f"  input  {port.any_name:12s} {port.partial_shape}")
        for port in ir.outputs:
            print(f"  output {port.any_name:12s} {port.partial_shape}")
        findings.append("OpenVINO 변환 OK")
    except Exception:
        traceback.print_exc()
        findings.append(
            "ONNX export 실패 — VITS의 동적 길이/확률적 길이예측 때문일 가능성이 큽니다."
        )

    print("\n\n=== 정리 " + "=" * 50)
    for line in findings:
        print(f"  - {line}")
    print(
        "\n어느 경우든 남는 질문 하나: 음소 변환이 파이썬 패키지라면,"
        "\novkit이 약속한 '변환하면 ovkit만으로 돈다'가 TTS에서는 성립하지 않습니다."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
