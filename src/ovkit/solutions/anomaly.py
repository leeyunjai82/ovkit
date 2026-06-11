"""Anomaly detection via anomalib's OpenVINO inferencer (optional extra).

INTERFACE STUB for v0. Planned default exposure: PatchCore / EfficientAD / PaDiM
through ``anomalib``'s ``OpenVINOInferencer`` (install ``ovkit[anomaly]``).
"""

from __future__ import annotations

from typing import Any


class AnomalyModel:
    """Wrapper around anomalib's OpenVINO inferencer (not yet implemented)."""

    def __init__(self, model_path: str, device: str = "AUTO", **kwargs: Any) -> None:
        raise NotImplementedError(
            "AnomalyModel is not implemented in v0. Planned: wrap anomalib "
            "OpenVINOInferencer (PatchCore/EfficientAD/PaDiM). Install 'ovkit[anomaly]'."
        )
