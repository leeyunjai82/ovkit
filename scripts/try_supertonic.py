#!/usr/bin/env python3
"""Run Supertonic through OpenVINO and listen to what comes out.

    python scripts/try_supertonic.py            # English and Korean
    python scripts/try_supertonic.py --steps 8

Two questions, one run.

**Does the port work?** The four graphs are chained here exactly as Supertone's
own ``py/helper.py`` chains them (MIT, read at
github.com/supertone-inc/supertonic) — the same preprocessing, the same latent
sizing, the same flow-matching loop — with OpenVINO in place of ONNX Runtime.
If English comes out as speech-shaped audio of the predicted length, the port
holds and ovkit's pipeline can be written from this file.

**Can it speak Korean?** ``unicode_indexer.json`` gives ids to 81 characters
and every one of them is ASCII: no 가-힣, no 자모. The repository card says
``language: en``. So the expectation is that Korean text arrives as a string of
unknown ids and produces nothing worth hearing — but an expectation is what got
the OCR alphabet wrong this morning, so it gets measured: how many characters
survive the table, how long the model says the sentence is, and how much energy
the waveform actually carries.

Writes the WAVs so a person can judge for themselves. Runs in Actions
(**Supertonic** workflow); this container cannot reach huggingface.co.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np

REPO = "Supertone/supertonic-3"

#: The graphs, by the role they play in the chain.
GRAPHS = {
    "duration": "onnx/duration_predictor.onnx",
    "text": "onnx/text_encoder.onnx",
    "vector": "onnx/vector_estimator.onnx",
    "vocoder": "onnx/vocoder.onnx",
}

#: What to say, in each language, so the two are comparable.
SENTENCES = (
    ("en", "Good morning. Today we are going to learn about machine learning."),
    ("ko", "좋은 아침입니다. 오늘은 기계 학습에 대해 배워 보겠습니다."),
    ("ko", "안녕하세요."),
)


# --- text -> ids (ported from py/helper.py, UnicodeProcessor) ---------------

_EMOJI = re.compile("[\U0001f300-\U0001faff☀-➿\U0001f1e6-\U0001f1ff]+", flags=re.UNICODE)
_REPLACE = {
    "–": "-",
    "‑": "-",
    "—": "-",
    "_": " ",
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "´": "'",
    "`": "'",
    "[": " ",
    "]": " ",
    "|": " ",
    "/": " ",
    "#": " ",
    "→": " ",
    "←": " ",
}


def preprocess(text: str, lang: str) -> str:
    """Normalise, then wrap in the language tag the model was trained with."""
    text = unicodedata.normalize("NFKD", text)
    text = _EMOJI.sub("", text)
    for old, new in _REPLACE.items():
        text = text.replace(old, new)
    text = re.sub(r"[♥☆♡©\\]", "", text)
    for old, new in {"@": " at ", "e.g.,": "for example, ", "i.e.,": "that is, "}.items():
        text = text.replace(old, new)
    for mark in (",", r"\.", "!", r"\?", ";", ":", "'"):
        text = re.sub(rf" {mark}", mark.replace("\\", ""), text)
    for pair in ('""', "''", "``"):
        while pair in text:
            text = text.replace(pair, pair[0])
    text = re.sub(r"\s+", " ", text).strip()
    if not re.search(r"[.!?;:,'\"')\]}…。」』】〉》›»]$", text):
        text += "."
    return f"<{lang}>{text}</{lang}>"


def to_ids(text: str, indexer: list[int]) -> tuple[np.ndarray, np.ndarray, int]:
    """Return ``(text_ids[1,L], text_mask[1,1,L], unknown_count)``.

    ``unknown_count`` is the finding: a character the table has no id for comes
    back as ``-1``, and a sentence made entirely of ``-1`` is a sentence the
    model was never taught to say.
    """
    codes = [ord(c) for c in text]
    ids = [indexer[c] if c < len(indexer) else -1 for c in codes]
    unknown = sum(1 for i in ids if i < 0)
    text_ids = np.asarray([ids], dtype=np.int64)
    text_mask = np.ones((1, 1, len(ids)), dtype=np.float32)
    return text_ids, text_mask, unknown


# --- the chain -------------------------------------------------------------


class Supertonic:
    """The four graphs, compiled by OpenVINO, chained as the reference does."""

    def __init__(self, files: dict[str, Path], cfg: dict[str, Any], indexer: list[int]):
        import openvino as ov

        core = ov.Core()
        self.nets = {role: core.compile_model(str(path), "CPU") for role, path in files.items()}
        self.sample_rate = int(cfg["ae"]["sample_rate"])
        self.base_chunk = int(cfg["ae"]["base_chunk_size"])
        self.compress = int(cfg["ttl"]["chunk_compress_factor"])
        self.latent_dim = int(cfg["ttl"]["latent_dim"]) * self.compress
        self.indexer = indexer

    def _run(self, role: str, feeds: dict[str, np.ndarray]) -> np.ndarray:
        net = self.nets[role]
        return net(feeds)[net.output(0)]

    def say(
        self, text: str, lang: str, style: dict[str, np.ndarray], steps: int, speed: float
    ) -> dict[str, Any]:
        prepared = preprocess(text, lang)
        text_ids, text_mask, unknown = to_ids(prepared, self.indexer)

        duration = self._run(
            "duration",
            {"text_ids": text_ids, "style_dp": style["dp"], "text_mask": text_mask},
        )
        duration = duration / speed
        text_emb = self._run(
            "text",
            {"text_ids": text_ids, "style_ttl": style["ttl"], "text_mask": text_mask},
        )

        # Latent sized from the predicted duration, exactly as sample_noisy_latent does.
        chunk = self.base_chunk * self.compress
        wav_len = int(duration.max() * self.sample_rate)
        latent_len = (wav_len + chunk - 1) // chunk
        latent_mask = np.ones((1, 1, latent_len), dtype=np.float32)
        rng = np.random.default_rng(0)  # seeded: two runs should sound the same
        latent = rng.standard_normal((1, self.latent_dim, latent_len)).astype(np.float32)
        latent *= latent_mask

        total = np.asarray([float(steps)], dtype=np.float32)
        for step in range(steps):
            latent = self._run(
                "vector",
                {
                    "noisy_latent": latent,
                    "text_emb": text_emb,
                    "style_ttl": style["ttl"],
                    "text_mask": text_mask,
                    "latent_mask": latent_mask,
                    "current_step": np.asarray([float(step)], dtype=np.float32),
                    "total_step": total,
                },
            )
        wav = self._run("vocoder", {"latent": latent})
        seconds = float(duration[0])
        trimmed = wav[0, : int(self.sample_rate * seconds)]
        return {
            "wav": trimmed,
            "seconds": seconds,
            "chars": len(prepared),
            "unknown": unknown,
            "rms": float(np.sqrt(np.mean(np.square(trimmed)))) if trimmed.size else 0.0,
            "peak": float(np.max(np.abs(trimmed))) if trimmed.size else 0.0,
        }


def load_style(path: Path) -> dict[str, np.ndarray]:
    """A voice style is two tensors stored as nested JSON lists."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for key, name in (("style_ttl", "ttl"), ("style_dp", "dp")):
        dims = raw[key]["dims"]
        flat = np.asarray(raw[key]["data"], dtype=np.float32).reshape(-1)
        out[name] = flat.reshape(1, int(dims[1]), int(dims[2]))
    return out


def main() -> int:
    global REPO

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=8, help="flow-matching 반복 횟수")
    parser.add_argument("--speed", type=float, default=1.05)
    parser.add_argument("--voice", default="M1")
    parser.add_argument("--repo", default=REPO, help="Hugging Face 저장소 id")
    parser.add_argument("--out", default="tts_out", help="WAV을 쓸 폴더")
    args = parser.parse_args()
    REPO = args.repo

    from huggingface_hub import hf_hub_download

    files = {role: Path(hf_hub_download(REPO, name)) for role, name in GRAPHS.items()}
    cfg = json.loads(Path(hf_hub_download(REPO, "onnx/tts.json")).read_text(encoding="utf-8"))
    indexer = json.loads(
        Path(hf_hub_download(REPO, "onnx/unicode_indexer.json")).read_text(encoding="utf-8")
    )
    style = load_style(Path(hf_hub_download(REPO, f"voice_styles/{args.voice}.json")))

    print(f"=== Supertonic을 OpenVINO로 돌린다 (voice={args.voice}, steps={args.steps})\n")
    started = time.perf_counter()
    tts = Supertonic(files, cfg, indexer)
    print(f"네 그래프 컴파일: {time.perf_counter() - started:.1f}초")
    print(f"샘플레이트 {tts.sample_rate} Hz, latent {tts.latent_dim}채널\n")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, (lang, sentence) in enumerate(SENTENCES):
        started = time.perf_counter()
        got = tts.say(sentence, lang, style, args.steps, args.speed)
        took = time.perf_counter() - started
        path = out_dir / f"{i}_{lang}.wav"
        _write_wav(path, got["wav"], tts.sample_rate)
        rows.append((lang, sentence, got, took, path))

    print(
        f"{'lang':5s} {'글자':>4s} {'모르는 글자':>10s} {'길이(초)':>9s} {'RMS':>8s} {'peak':>7s}"
    )
    for lang, sentence, got, took, path in rows:
        share = got["unknown"] / got["chars"] * 100 if got["chars"] else 0
        print(
            f"{lang:5s} {got['chars']:4d} {got['unknown']:6d} ({share:3.0f}%) "
            f"{got['seconds']:9.2f} {got['rms']:8.4f} {got['peak']:7.3f}   "
            f"{took:5.2f}초 걸림 -> {path}"
        )
        print(f"      {sentence}")

    print(
        "\n읽는 법: '모르는 글자'는 unicode_indexer에 id가 없어 -1로 들어간 글자 수다."
        "\n그 비율이 100%에 가까우면 모델은 문장을 전혀 받지 못한 것이고,"
        "\n그때 나오는 소리는 문장과 아무 상관이 없다. WAV은 아티팩트로 올라가니"
        "\n직접 들어 보고 판단하면 된다."
    )
    return 0


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    """16-bit mono WAV, without pulling in a dependency for four lines."""
    import wave

    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sample_rate)
        fh.writeframes(pcm.tobytes())


if __name__ == "__main__":
    raise SystemExit(main())
