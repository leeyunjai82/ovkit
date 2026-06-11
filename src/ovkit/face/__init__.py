"""Face analysis (detection, landmarks, recognition, head pose, emotion, ...).

INTERFACE STUB for v0. Models will come from the ovkit HF mirror
(``leeyunjai/ovkit-models``), which hosts OMZ-derived **Apache-2.0** IR:
face-detection-retail-0005, landmarks-regression-retail-0009,
face-reidentification-retail-0095, head-pose-estimation-adas-0001,
emotions-recognition-retail-0003, age-gender-recognition-retail-0013, and an
anti-spoofing model.

License note (spec §7): InsightFace (SCRFD/ArcFace) **pretrained weights** are
non-commercial and MUST NOT be bundled or downloaded; architecture reference
only.
"""

from __future__ import annotations

from .analyzer import FaceAnalyzer

__all__ = ["FaceAnalyzer"]
