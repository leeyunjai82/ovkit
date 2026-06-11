"""Task auto-detection tests using lightweight fake backends."""

from __future__ import annotations

import pytest

from ovkit.core.errors import TaskDetectionError
from ovkit.core.tasks import detect_task


class _FakeBackend:
    def __init__(self, sigs, rt=None):
        self._sigs = sigs
        self._rt = rt or {}

    def output_signatures(self):
        return self._sigs

    def rt_info(self, *keys):
        return self._rt.get(keys)


def test_override_wins():
    be = _FakeBackend([("probs", (1, 1000))])
    assert detect_task(be, manifest_task="classify", override="detect") == "detect"


def test_manifest_task_used():
    be = _FakeBackend([("x", (1, 1000))])
    assert detect_task(be, manifest_task="segment") == "segment"


def test_rt_info_task():
    be = _FakeBackend([("x", (1, 7, 8))], rt={("model_info", "task"): "pose estimation"})
    assert detect_task(be) == "pose"


def test_signature_detr_detect():
    be = _FakeBackend([("logits", (1, 300, 80)), ("boxes", (1, 300, 4))])
    assert detect_task(be) == "detect"


def test_signature_classify():
    be = _FakeBackend([("probs", (1, 1000))])
    assert detect_task(be) == "classify"


def test_ambiguous_raises():
    be = _FakeBackend([("weird", (1, 2, 3, 4, 5))])
    with pytest.raises(TaskDetectionError):
        detect_task(be)
