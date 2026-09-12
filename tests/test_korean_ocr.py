"""Korean text recognition: the decoder, the charset, and the language switch.

The failure mode here is silent. A wrong blank index or an off-by-one charset
does not raise — it returns ``""`` or gibberish, and a photo of a sign comes
back empty with nothing to explain why. So the decode is tested against a
hand-built alphabet where the right answer is known.
"""

from __future__ import annotations

import numpy as np
import pytest

from ovkit.core import registry
from ovkit.pipelines.text import TextReader
from ovkit.recognize.ocr import OCRAdapter

#: A four-symbol stand-in for PaddleOCR's 3,700-line korean_dict.txt.
ALPHABET = ["안", "녕", "하", "세"]


def _logits(class_ids: list[int], num_classes: int) -> np.ndarray:
    """One-hot ``[T, 1, C]`` logits that argmax to ``class_ids``."""
    a = np.zeros((len(class_ids), 1, num_classes), dtype=np.float32)
    for t, c in enumerate(class_ids):
        a[t, 0, c] = 10.0
    return a


def _paddle_adapter(**post) -> OCRAdapter:
    settings = {"charset": "labels", "blank_first": True, **post}
    return OCRAdapter(
        postprocess=settings,
        names={i: c for i, c in enumerate(ALPHABET)},
    )


def test_paddle_numbering_reads_the_word():
    """Blank is class 0, so dictionary entry i is class i + 1."""
    adapter = _paddle_adapter()
    # blank, 안(1), 안(repeat), blank, 녕(2)  ->  "안녕"
    text = adapter._ctc_greedy(_logits([0, 1, 1, 0, 2], len(ALPHABET) + 1))
    assert text == "안녕"


def test_the_blank_is_not_printed():
    adapter = _paddle_adapter()
    assert adapter._ctc_greedy(_logits([0, 0, 0], len(ALPHABET) + 1)) == ""


def test_space_rides_at_the_end_of_the_table():
    """PaddleOCR appends the space in code, so it is not a line in the file."""
    adapter = _paddle_adapter(space=True)
    space_id = len(ALPHABET) + 1  # blank + 4 symbols -> space is the next class
    text = adapter._ctc_greedy(_logits([1, space_id, 3], len(ALPHABET) + 2))
    assert text == "안 하"


def test_without_space_the_table_stops_at_the_dictionary():
    adapter = _paddle_adapter()
    symbols, blank = adapter._symbols()
    assert blank == 0
    assert symbols == ["", *ALPHABET]


def test_the_latin_default_still_decodes():
    """The OMZ models kept their built-in alphabet — blank last, no labels."""
    adapter = OCRAdapter(postprocess={})
    symbols, blank = adapter._symbols()
    assert blank == len(symbols) - 1 and symbols[blank] == "#"
    assert adapter._ctc_greedy(_logits([0, 0, 1, blank, 1], len(symbols))) == "011"


def test_the_model_is_registered_against_the_mirror():
    entry = registry.resolve("text_recognition_ko")
    assert entry is not None and entry.name == "korean_text_recognition"
    assert entry.repo == "leeyunjai/ovkit-models"
    assert entry.postprocess["charset"] == "labels"
    assert entry.postprocess["blank_first"] is True


@pytest.mark.parametrize(
    ("language", "expected"),
    [("ko", "text_recognition_ko"), ("en", "text_recognition")],
)
def test_read_text_picks_the_recogniser_for_the_language(monkeypatch, language, expected):
    """A Korean sign read with the Latin model comes back empty, not wrong."""
    monkeypatch.setenv("OVKIT_LANG", language)
    assert TextReader().recognizer == expected


def test_an_explicit_recogniser_wins(monkeypatch):
    monkeypatch.setenv("OVKIT_LANG", "ko")
    assert TextReader(recognizer="text_recognition").recognizer == "text_recognition"


def test_a_missing_korean_model_falls_back_out_loud(monkeypatch):
    """Not on the mirror yet, or offline: say so once, then read what we can."""
    monkeypatch.setenv("OVKIT_LANG", "ko")
    reader = TextReader()

    calls: list[str] = []

    class _Reader:
        def __call__(self, _crop):
            return []

    def fake_model(name):
        calls.append(name)
        if name == "text_recognition_ko":
            raise RuntimeError("not on the mirror")
        return _Reader()

    monkeypatch.setattr(reader, "model", fake_model)
    with pytest.warns(RuntimeWarning, match="0-9 and a-z"):
        assert reader._read(np.zeros((8, 8, 3), np.uint8)) == ""
    assert calls == ["text_recognition_ko", "text_recognition"]
    assert reader.recognizer == "text_recognition", "it should not warn twice"


def test_the_space_token_decodes_to_a_space():
    """A space cannot survive as a line in a labels file, so it travels as a token.

    The alternative — dropping the line — shifts every class after it by one,
    and the model reads fluent nonsense without raising anything.
    """
    adapter = OCRAdapter(
        postprocess={"charset": "labels", "blank_first": True},
        names={0: "안", 1: "<space>", 2: "녕"},
    )
    symbols, blank = adapter._symbols()
    assert symbols == ["", "안", " ", "녕"] and blank == 0
    assert adapter._ctc_greedy(_logits([1, 2, 3], 4)) == "안 녕"
