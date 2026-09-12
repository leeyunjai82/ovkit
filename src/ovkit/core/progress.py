"""What ovkit says while it is busy.

The first call a beginner makes downloads a model, converts it, and compiles
it — tens of seconds during which ovkit said, until now, absolutely nothing.
A blank terminal is indistinguishable from a hang, and the first thing someone
learns about the library is that it might be broken.

So: say what is happening, say it once, and say how long it will keep
happening. Everything here goes to **stderr**, so piping ``ovkit run`` into a
file still gets only results, and everything obeys ``$OVKIT_QUIET``.

    [ovkit] 'detect' 모델을 처음 받습니다 — rtdetr_r18, 약 81 MB
    [ovkit] 한 번만 받으면 됩니다. 다음부터는 바로 시작합니다.
    [ovkit] 준비 끝 (12.4초). 이제 돌립니다.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager

from .i18n import lang

#: Set to ``1`` to silence every message in this module.
ENV_QUIET = "OVKIT_QUIET"


def quiet() -> bool:
    """True when the user asked for silence."""
    return os.environ.get(ENV_QUIET, "").strip() in {"1", "true", "True", "yes"}


def say(korean: str, english: str) -> None:
    """One line on stderr, in the display language."""
    if quiet():
        return
    print(f"[ovkit] {korean if lang() == 'ko' else english}", file=sys.stderr, flush=True)


def human_bytes(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:,.0f} {unit}" if unit == "B" else f"{size:,.1f} {unit}"
        size /= 1024
    return f"{size:,.1f} GB"


def _short(description: str) -> str:
    """The first clause of a model's description — the rest is for the catalog."""
    head = description.split("—")[0].split(". ")[0].strip(" .")
    return head[:60]


def downloading(name: str, what: str = "", size: int | None = None) -> None:
    """Announce a download before the wait, not after it."""
    detail = _short(what) or name
    if size:
        detail = f"{detail}, {human_bytes(size)}"
    say(
        f"'{name}' 모델을 처음 받습니다 — {detail}",
        f"downloading '{name}' for the first time — {detail}",
    )
    say(
        "한 번만 받으면 됩니다. 다음부터는 바로 시작합니다.",
        "this happens once; afterwards it starts immediately.",
    )


def converting(name: str) -> None:
    say(
        f"'{name}'을(를) OpenVINO 형식으로 바꾸는 중입니다 (한 번만).",
        f"converting '{name}' to OpenVINO IR (once).",
    )


@contextmanager
def step(korean: str, english: str) -> Iterator[None]:
    """Announce a slow step and report how long it took."""
    started = time.perf_counter()
    say(korean, english)
    try:
        yield
    finally:
        seconds = time.perf_counter() - started
        if seconds >= 1.0:
            say(f"끝났습니다 ({seconds:.1f}초).", f"done ({seconds:.1f}s).")


class Bar:
    """A one-line download meter for sources that stream bytes themselves.

    ``huggingface_hub`` draws its own; a plain URL fetch had nothing, so a
    300 MB file looked exactly like a frozen process.
    """

    def __init__(self, name: str, total: int | None) -> None:
        self.name = name
        self.total = int(total or 0)
        self.seen = 0
        self._last = 0.0
        self._on = not quiet() and sys.stderr.isatty()

    def advance(self, chunk: int) -> None:
        self.seen += chunk
        if not self._on:
            return
        now = time.perf_counter()
        if now - self._last < 0.2 and self.seen != self.total:
            return
        self._last = now
        if self.total:
            share = self.seen / self.total
            filled = int(share * 24)
            bar = "#" * filled + "." * (24 - filled)
            tail = f"{share * 100:5.1f}%  {human_bytes(self.seen)} / {human_bytes(self.total)}"
        else:
            bar = "." * 24
            tail = human_bytes(self.seen)
        print(f"\r[ovkit] {self.name[:28]:28s} {bar} {tail}", end="", file=sys.stderr, flush=True)

    def close(self) -> None:
        if self._on:
            print(file=sys.stderr, flush=True)
