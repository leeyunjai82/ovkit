"""Higher-level solutions assembled from primitives: anomaly, OCR, tracking, ReID.

v0 ships interface stubs only; the structure (compose on top of detect/recognize)
is fixed so the implementations can be filled in without API churn.
"""

from __future__ import annotations

__all__ = ["anomaly", "ocr", "tracking", "reid"]
