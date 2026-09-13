#!/usr/bin/env python3
"""Draw the README's terminal demo as an animated GIF.

    python scripts/make_demo_gif.py --out docs/_static/demo.gif

This is a **rendering of a real session**, not a screen capture. Every line it
types and every line it prints was produced by ovkit on an Intel Core Ultra 7
255H laptop; the script only redraws them at a readable pace. Nothing here is
invented — if a number changes, re-run the commands and edit ``SCRIPT`` to
match, rather than editing the number to look better.

A screen recording of the real thing is better than this and should replace it
when there is one. This exists so the README is not blank while waiting.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# --- look -------------------------------------------------------------------

W, H = 760, 420
BG = (18, 22, 29)  # matches the ovkit GUI
CHROME = (26, 32, 41)
FG = (230, 237, 243)
MUTED = (139, 148, 158)
BRAND = (34, 184, 207)
GREEN = (63, 185, 80)
PROMPT = (166, 226, 46)

#: Monospace for the terminal, and a CJK face for the Hangul the Latin font has
#: no glyphs for. Rendered per character, so a line can mix the two.
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
CJK = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"

SIZE = 15
LINE_H = 24
PAD_X, PAD_Y = 18, 52

#: Frames per second the GIF plays at, and how many frames each beat lasts.
FPS = 20
TYPE_FRAMES = 1  # per character
PAUSE_AFTER_COMMAND = 6
PAUSE_AFTER_OUTPUT = 4
HOLD_AT_END = 40


#: ``("cmd", text)`` is typed at a prompt; ``("out", text)`` appears at once.
#:
#: Only output this project has actually seen. `ovkit devices` and
#: `examples/speak.py` are from the 255H laptop; the `Model(...)` lines are the
#: README's own examples.
SCRIPT: tuple[tuple[str, str], ...] = (
    ("cmd", "pip install ovkit"),
    ("out", "Successfully installed ovkit-0.4.1"),
    ("gap", ""),
    ("cmd", "ovkit devices"),
    ("out", "CPU"),
    ("out", "GPU"),
    ("out", "NPU"),
    ("gap", ""),
    ("cmd", "python examples/speak.py"),
    ("out", "4.7초 · 44100 Hz · 목소리 F1"),
    ("out", "-> speak.wav"),
    ("out", "-> speak.png (파형)"),
)

#: Drawn under the terminal, as a caption rather than as fake terminal output.
CAPTION = 'Model("읽어주기", "안녕하세요") — 20 capabilities, one call each'


def _fonts() -> tuple[ImageFont.FreeTypeFont, ImageFont.FreeTypeFont, ImageFont.FreeTypeFont]:
    mono = ImageFont.truetype(MONO, SIZE)
    cjk = ImageFont.truetype(CJK, SIZE)
    small = ImageFont.truetype(MONO, 13)
    return mono, cjk, small


def _needs_cjk(ch: str) -> bool:
    """True for characters the Latin monospace font has no glyph for.

    Deliberately not "is this wide": the middle dot ovkit separates fields with
    (``·``) is in DejaVu and looked absurd when this treated it as CJK and gave
    it two cells.
    """
    cp = ord(ch)
    return (
        0x1100 <= cp <= 0x11FF
        or 0x3000 <= cp <= 0x9FFF
        or 0xAC00 <= cp <= 0xD7A3
        or 0xFF00 <= cp <= 0xFF60
    )


def _draw_text(draw: ImageDraw.ImageDraw, xy, text, mono, cjk, fill):
    """Draw a string one character at a time, switching font where needed.

    Advances by each glyph's own width rather than by a fixed cell. Assuming
    "CJK is two cells" spaced Hangul out into 목 소 리 — the Latin font's cell
    and this face's glyphs are not in that ratio.
    """
    x, y = xy
    for ch in text:
        font = cjk if _needs_cjk(ch) else mono
        draw.text((x, y - 1 if font is cjk else y), ch, font=font, fill=fill)
        x += font.getlength(ch)
    return x


def _chrome(img: Image.Image, mono, small) -> None:
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, W, 36], fill=CHROME)
    for i, colour in enumerate(((255, 95, 86), (255, 189, 46), (39, 201, 63))):
        draw.ellipse([18 + i * 20, 13, 28 + i * 20, 23], fill=colour)
    draw.text((W // 2 - 30, 10), "ovkit", font=small, fill=MUTED)


def _frame(lines, mono, cjk, small) -> Image.Image:
    """One frame: the chrome, the lines so far, and the caption."""
    img = Image.new("RGB", (W, H), BG)
    _chrome(img, mono, small)
    draw = ImageDraw.Draw(img)

    y = PAD_Y
    for kind, text, done in lines:
        if kind == "gap":
            y += LINE_H // 2
            continue
        x = PAD_X
        if kind == "cmd":
            x = _draw_text(draw, (x, y), "$ ", mono, cjk, PROMPT)
            x = _draw_text(draw, (x, y), text, mono, cjk, FG)
            if not done:  # a cursor, while it is still being typed
                draw.rectangle([x, y + 2, x + mono.getlength("M") - 2, y + SIZE + 4], fill=BRAND)
        else:
            colour = GREEN if text in ("CPU", "GPU", "NPU") else MUTED
            _draw_text(draw, (x + 16, y), text, mono, cjk, colour)
        y += LINE_H

    draw.line([0, H - 42, W, H - 42], fill=CHROME, width=1)
    _draw_text(draw, (PAD_X, H - 30), CAPTION, small, cjk, MUTED)
    return img


def build() -> list[Image.Image]:
    mono, cjk, small = _fonts()
    frames: list[Image.Image] = []
    lines: list[tuple[str, str, bool]] = []

    for kind, text in SCRIPT:
        if kind == "gap":
            lines.append(("gap", "", True))
            continue
        if kind == "cmd":
            lines.append(("cmd", "", False))
            for i in range(1, len(text) + 1):
                lines[-1] = ("cmd", text[:i], False)
                frames.extend([_frame(lines, mono, cjk, small)] * TYPE_FRAMES)
            lines[-1] = ("cmd", text, True)
            frames.extend([_frame(lines, mono, cjk, small)] * PAUSE_AFTER_COMMAND)
        else:
            lines.append(("out", text, True))
            frames.extend([_frame(lines, mono, cjk, small)] * PAUSE_AFTER_OUTPUT)

    frames.extend([frames[-1]] * HOLD_AT_END)
    return frames


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="docs/_static/demo.gif")
    args = parser.parse_args()

    frames = build()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # A flat palette: this is text on one background, so 32 colours is plenty
    # and keeps the file small enough for a README to load instantly.
    flat = [f.convert("P", palette=Image.ADAPTIVE, colors=32) for f in frames]
    flat[0].save(
        out,
        save_all=True,
        append_images=flat[1:],
        duration=int(1000 / FPS),
        loop=0,
        optimize=True,
    )
    size = out.stat().st_size
    print(f"{out}  {len(frames)} frames  {size / 1024:,.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
