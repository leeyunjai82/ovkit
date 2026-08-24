"""Caption rendering: one readable overlay, never two overprinted ones."""

from __future__ import annotations

import cv2
import numpy as np

from ovkit.core.results import Boxes, Probs, Results, draw_caption, wrap_text

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _width(text: str, scale: float = 0.6) -> int:
    return cv2.getTextSize(text, FONT, scale, 1)[0][0]


def test_wrap_text_keeps_every_line_within_the_width():
    text = "male 0.99 · long pants 0.98 · long sleeves 0.95 · coat/jacket 0.71 · bag 0.66"
    lines = wrap_text(text, 200, 0.6)
    assert len(lines) > 1
    assert all(_width(line) <= 200 for line in lines)
    assert " ".join(lines).split() == text.split()  # no words lost


def test_wrap_text_splits_a_word_too_long_to_fit():
    lines = wrap_text("A" * 200, 120, 0.6)
    assert len(lines) > 1
    assert all(_width(line) <= 120 for line in lines)


def test_draw_caption_darkens_only_the_band_it_uses():
    img = np.full((240, 320, 3), 200, np.uint8)
    out = draw_caption(img.copy(), "closed 0.73", 0.6)
    assert out.shape == img.shape
    assert out[:20].mean() < img[:20].mean()  # band darkened
    assert np.array_equal(out[-40:], img[-40:])  # rest untouched


def test_draw_caption_clips_a_very_long_answer():
    img = np.full((240, 320, 3), 200, np.uint8)
    out = draw_caption(img.copy(), "word " * 400, 0.6, max_lines=3)
    assert out.shape == img.shape
    assert np.array_equal(out[150:], img[150:])  # never fills the whole frame


def test_plot_draws_one_caption_not_two():
    """probs + text used to be drawn at y=30 and y=36, overprinting each other."""
    img = np.full((240, 320, 3), 200, np.uint8)
    r = Results(img, task="classify", names={0: "open", 1: "closed"}, probs=Probs([0.27, 0.73]))
    r.text = "closed 0.73"
    both = r.plot()

    text_only = Results(img, task="classify", names={0: "open", 1: "closed"})
    text_only.text = "closed 0.73"
    assert np.array_equal(both, text_only.plot())


def test_plot_caption_has_no_stray_quotes():
    img = np.zeros((120, 400, 3), np.uint8)
    r = Results(img, task="generic")
    r.text = "x"
    # The old overlay wrapped the answer in quotes; the summary is the answer.
    assert r.summary() == "x"


def test_box_label_stays_inside_the_frame():
    img = np.full((120, 200, 3), 30, np.uint8)
    boxes = Boxes(np.array([[0, 0, 60, 40, 0.9, 0]], np.float32))
    r = Results(img, task="detect", names={0: "person"}, boxes=boxes)
    out = r.plot(caption=False)
    # A box flush with the top edge still gets a visible label bar.
    assert out[:14, :40].max() > 60
