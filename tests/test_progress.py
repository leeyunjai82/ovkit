"""What ovkit says while it is busy — and that it can be told to stop.

The first call downloads a model, converts it and compiles it. Until this
existed, all of that happened in silence: a beginner's first experience of the
library was a terminal that might be broken. These tests cover the two things
that make the messages safe to have — they go to stderr (so piped results stay
clean) and `$OVKIT_QUIET` silences them.
"""

from __future__ import annotations

import pytest

from ovkit.core import progress


def test_it_speaks_on_stderr_not_stdout(capsys, monkeypatch):
    monkeypatch.delenv("OVKIT_QUIET", raising=False)
    monkeypatch.setenv("OVKIT_LANG", "en")
    progress.say("안녕", "hello")
    captured = capsys.readouterr()
    assert captured.out == "", "a message on stdout would end up in piped results"
    assert "hello" in captured.err


def test_quiet_silences_everything(capsys, monkeypatch):
    monkeypatch.setenv("OVKIT_QUIET", "1")
    progress.say("안녕", "hello")
    progress.downloading("detect", "rtdetr_r18", 81_000_000)
    progress.converting("detect")
    assert capsys.readouterr().err == ""


def test_the_download_message_says_what_and_how_big(capsys, monkeypatch):
    monkeypatch.delenv("OVKIT_QUIET", raising=False)
    monkeypatch.setenv("OVKIT_LANG", "en")
    progress.downloading("detect", "RT-DETR R18", 81_000_000)
    err = capsys.readouterr().err
    assert "detect" in err and "RT-DETR R18" in err and "MB" in err
    assert "once" in err, "the wait is bearable if you know it happens once"


def test_korean_is_the_default_voice(capsys, monkeypatch):
    monkeypatch.delenv("OVKIT_QUIET", raising=False)
    monkeypatch.delenv("OVKIT_LANG", raising=False)
    progress.converting("detect")
    assert "바꾸는 중" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("size", "shown"),
    [(512, "512 B"), (81_000_000, "77.2 MB"), (2_000_000_000, "1.9 GB")],
)
def test_sizes_read_as_sizes(size, shown):
    assert progress.human_bytes(size) == shown


def test_the_meter_stays_off_when_nobody_is_watching(capsys, monkeypatch):
    """Not a terminal (a log file, a notebook, CI) means no carriage returns."""
    monkeypatch.delenv("OVKIT_QUIET", raising=False)
    bar = progress.Bar("model.bin", 1000)
    bar.advance(500)
    bar.close()
    assert "\r" not in capsys.readouterr().err
