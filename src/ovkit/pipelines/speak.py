"""Read a sentence out loud.

    from ovkit import Model

    r = Model("읽어주기", "안녕하세요. 오늘은 기계 학습을 배웁니다.")
    r.save("hello.wav")

Four networks, not one. Supertonic-3 predicts how long the sentence takes,
encodes the text, denoises a latent towards that text a few times, and turns
the latent into a waveform. The chain is Supertone's — ported from their own
``py/helper.py`` (MIT) and confirmed against the real graphs before a line of
this file was written, because four interfaces guessed is four times the
mistake one guessed interface already cost this project.

31 languages, Korean among them. ovkit picks the language from the text unless
you name one.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from ..core.results import Results
from .base import Pipeline

#: Where the mirrored model's tables and voices live.
_REPO = "leeyunjai/ovkit-models"
_DATA = "tts/supertonic3/data"
_GROUP = "supertonic3"

#: The ten preset voices Supertone ships, F1..F5 and M1..M5.
VOICES: tuple[str, ...] = tuple(f"{sex}{n}" for sex in "FM" for n in range(1, 6))

#: Languages the model was trained on (from its model card). The language is
#: not decoration: the text is wrapped in ``<ko>…</ko>`` before it is indexed,
#: and the tag changes how the same letters are pronounced.
LANGUAGES: frozenset[str] = frozenset(
    "en ko ja ar bg cs da de el es et fi fr hi hr hu id it lt lv nl pl pt ro "
    "ru sk sl sv tr uk vi".split()
)

#: How many flow-matching steps to run. Supertone's own examples use 8; their
#: published timings go down to 2. More steps is cleaner and slower, and the
#: difference is audible long before it is expensive — a sentence takes well
#: under a second on a laptop CPU either way.
DEFAULT_STEPS = 8

#: Slightly faster than the model's natural pace, which is Supertone's own
#: default and sounds less sleepy read aloud in a classroom.
DEFAULT_SPEED = 1.05

_HANGUL = re.compile(r"[가-힣ᄀ-ᇿ㄰-㆏]")
_KANA = re.compile(r"[぀-ヿ]")
_HAN = re.compile(r"[一-鿿]")


def detect_language(text: str) -> str:
    """Guess the language from the script the text is written in.

    Only what the script can actually tell you. Hangul is Korean and kana is
    Japanese, but Latin letters are thirty of these languages at once, so
    anything else answers ``"en"`` — and a caller who knows better passes
    ``lang=`` and is believed.
    """
    if _HANGUL.search(text):
        return "ko"
    if _KANA.search(text):
        return "ja"
    if _HAN.search(text):
        return "ja"
    return "en"


class Speaker(Pipeline):
    """Turn text into speech — ``Model("읽어주기", "안녕하세요")``."""

    name = "speak"
    description = "글을 사람 목소리로 읽어 줍니다 (한국어·영어 등 31개 언어)"

    def __init__(
        self,
        device: str = "AUTO",
        voice: str = "F1",
        steps: int = DEFAULT_STEPS,
        speed: float = DEFAULT_SPEED,
        lang: str | None = None,
        seed: int | None = 0,
    ) -> None:
        super().__init__(device=device)
        if voice not in VOICES:
            raise ValueError(f"목소리는 {', '.join(VOICES)} 중 하나입니다 (받은 값: {voice!r}).")
        if lang is not None and lang not in LANGUAGES:
            known = ", ".join(sorted(LANGUAGES))
            raise ValueError(f"'{lang}'는 지원하지 않는 언어입니다. 가능한 값: {known}.")
        self.voice = voice
        self.steps = max(1, int(steps))
        self.speed = float(speed)
        self.lang = lang
        #: Seeded by default: the same sentence should sound the same twice.
        #: Pass ``seed=None`` for a fresh sample each call.
        self.seed = seed
        self._tables: dict[str, Any] = {}
        self._styles: dict[str, dict[str, np.ndarray]] = {}

    # -- the data that makes the graphs usable ------------------------------

    def _table(self, filename: str) -> Any:
        if filename not in self._tables:
            from ..core.download import fetch_data

            path = fetch_data(_REPO, f"{_DATA}/{filename}", group=_GROUP)
            self._tables[filename] = json.loads(path.read_text(encoding="utf-8"))
        return self._tables[filename]

    def _style(self, voice: str) -> dict[str, np.ndarray]:
        """A voice is two small tensors, stored as nested JSON lists."""
        if voice not in self._styles:
            from ..core.download import fetch_data

            path = fetch_data(_REPO, f"{_DATA}/voices/{voice}.json", group=_GROUP)
            raw = json.loads(path.read_text(encoding="utf-8"))
            style = {}
            for key, short in (("style_ttl", "ttl"), ("style_dp", "dp")):
                dims = raw[key]["dims"]
                flat = np.asarray(raw[key]["data"], dtype=np.float32).reshape(-1)
                style[short] = flat.reshape(1, int(dims[1]), int(dims[2]))
            self._styles[voice] = style
        return self._styles[voice]

    # -- text -> ids --------------------------------------------------------

    def _ids(self, text: str, lang: str) -> tuple[np.ndarray, np.ndarray, int]:
        """Return ``(text_ids, text_mask, unknown)`` for one prepared sentence.

        ``unknown`` counts characters the table has no id for. They are not an
        error — ``<ko>`` itself contributes a few — but a sentence that is
        *mostly* unknown is a sentence the model never received, and the caller
        deserves to be told rather than handed noise.
        """
        indexer = self._table("unicode_indexer.json")
        prepared = _prepare(text, lang)
        ids = [indexer[ord(c)] if ord(c) < len(indexer) else -1 for c in prepared]
        unknown = sum(1 for i in ids if i < 0)
        text_ids = np.asarray([ids], dtype=np.int64)
        text_mask = np.ones((1, 1, len(ids)), dtype=np.float32)
        return text_ids, text_mask, unknown

    # -- running ------------------------------------------------------------

    def say(
        self,
        text: str,
        *,
        lang: str | None = None,
        voice: str | None = None,
        steps: int | None = None,
        speed: float | None = None,
    ) -> Results:
        """Synthesise one sentence and return it as a :class:`Results`."""
        import warnings

        from ..audio import waveform

        text = str(text).strip()
        if not text:
            raise ValueError("읽을 글이 비어 있습니다.")
        lang = lang or self.lang or detect_language(text)
        voice = voice or self.voice
        steps = self.steps if steps is None else max(1, int(steps))
        speed = self.speed if speed is None else float(speed)

        style = self._style(voice)
        text_ids, text_mask, unknown = self._ids(text, lang)
        if unknown > len(text) // 2:
            warnings.warn(
                f"'{lang}'로 읽으려는 글자 대부분({unknown}/{text_ids.shape[1]})이 "
                f"이 모델의 글자표에 없습니다. 나오는 소리는 글과 무관할 수 있습니다.",
                RuntimeWarning,
                stacklevel=2,
            )

        duration = self.model("supertonic3_duration_predictor").infer(
            {"text_ids": text_ids, "style_dp": style["dp"], "text_mask": text_mask}
        )
        seconds = float(np.asarray(_first(duration)).reshape(-1)[0]) / speed

        text_emb = _first(
            self.model("supertonic3_text_encoder").infer(
                {"text_ids": text_ids, "style_ttl": style["ttl"], "text_mask": text_mask}
            )
        )

        cfg = self._table("tts.json")
        sample_rate = int(cfg["ae"]["sample_rate"])
        compress = int(cfg["ttl"]["chunk_compress_factor"])
        chunk = int(cfg["ae"]["base_chunk_size"]) * compress
        latent_dim = int(cfg["ttl"]["latent_dim"]) * compress
        latent_len = max(1, (int(seconds * sample_rate) + chunk - 1) // chunk)

        rng = np.random.default_rng(self.seed)
        latent = rng.standard_normal((1, latent_dim, latent_len)).astype(np.float32)
        latent_mask = np.ones((1, 1, latent_len), dtype=np.float32)

        # Flow matching: each pass hands its output back as the next input.
        # The graph is named `denoised_latent`, and it is the updated latent —
        # not the vector field, which would need integrating here.
        estimator = self.model("supertonic3_vector_estimator")
        total = np.asarray([float(steps)], dtype=np.float32)
        for step in range(steps):
            latent = _first(
                estimator.infer(
                    {
                        "noisy_latent": latent,
                        "text_emb": text_emb,
                        "style_ttl": style["ttl"],
                        "text_mask": text_mask,
                        "latent_mask": latent_mask,
                        "current_step": np.asarray([float(step)], dtype=np.float32),
                        "total_step": total,
                    }
                )
            )

        wav = _first(self.model("supertonic3_vocoder").infer({"latent": latent}))
        samples = np.asarray(wav, dtype=np.float32).reshape(-1)[: int(seconds * sample_rate)]

        result = Results(waveform(samples, sample_rate), task=self.name)
        result.audio = (samples, sample_rate)
        result.text = text
        result.labels = [f"{lang} · {voice} · {seconds:.1f}초"]
        return result

    def run(self, image: Any, **kwargs: Any) -> Results:  # pragma: no cover - not an image task
        raise TypeError(
            "'speak'은 그림이 아니라 글을 받습니다. Model('읽어주기', '안녕하세요')처럼 쓰세요."
        )

    def predict(self, source: Any, *, stream: bool = False, **kwargs: Any) -> Any:
        """Speak ``source``: a sentence, a list of them, or a ``.txt`` file."""
        texts = _as_texts(source)
        results = [self.say(t, **kwargs) for t in texts]
        for r in results:
            r.device = r.device or self.device
        if stream:
            return iter(results)
        return results


# --- helpers ---------------------------------------------------------------


def _first(outputs: Any) -> np.ndarray:
    """The single output of a one-output graph, whatever it is keyed by."""
    if isinstance(outputs, dict):
        return next(iter(outputs.values()))
    return outputs


def _as_texts(source: Any) -> list[str]:
    if isinstance(source, (list, tuple)):
        return [t for item in source for t in _as_texts(item)]
    text = str(source)
    # A path to a text file is a perfectly ordinary thing to want read aloud,
    # and `Path(a whole sentence).is_file()` is False, so this is safe to try.
    try:
        path = Path(text)
        if path.suffix.lower() in {".txt", ".md"} and path.is_file():
            return [path.read_text(encoding="utf-8")]
    except OSError:  # a "path" longer than the filesystem allows
        pass
    return [text]


_EMOJI = re.compile("[\U0001f300-\U0001faff☀-➿\U0001f1e6-\U0001f1ff]+", flags=re.UNICODE)

#: Characters the model's table does not have, mapped to ones it does. From
#: Supertone's own preprocessing — a curly quote is silence otherwise.
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

_ENDS_SENTENCE = re.compile(r"[.!?;:,'\"')\]}…。」』】〉》›»]$")


def _prepare(text: str, lang: str) -> str:
    """Normalise a sentence and wrap it in its language tag.

    Ported from Supertone's ``UnicodeProcessor._preprocess_text``. The NFKD
    normalisation matters more than it looks: it splits each Hangul syllable
    into its jamo, which is the form the character table is indexed by.
    """
    import unicodedata

    text = unicodedata.normalize("NFKD", text)
    text = _EMOJI.sub("", text)
    for old, new in _REPLACE.items():
        text = text.replace(old, new)
    text = re.sub(r"[♥☆♡©\\]", "", text)
    for old, new in (("@", " at "), ("e.g.,", "for example, "), ("i.e.,", "that is, ")):
        text = text.replace(old, new)
    for mark in (",", ".", "!", "?", ";", ":", "'"):
        text = text.replace(f" {mark}", mark)
    for pair in ('""', "''", "``"):
        while pair in text:
            text = text.replace(pair, pair[0])
    text = re.sub(r"\s+", " ", text).strip()
    if not _ENDS_SENTENCE.search(text):
        text += "."
    return f"<{lang}>{text}</{lang}>"
