"""Anomaly detection through anomalib's OpenVINO inferencer.

Anomaly models are trained per production line, so ovkit serves no weights for
them — you point this at the ``.xml`` anomalib exported and get the same
:class:`~ovkit.Results` every other ovkit call returns::

    from ovkit.solutions import AnomalyModel

    model = AnomalyModel("patchcore/weights/openvino/model.xml")
    r = model("part_0142.png")[0]
    print(r.summary())        # 'anomaly 0.87 (threshold 0.50)'
    r.save("defect.jpg")      # the heat map over the part

Needs ``pip install "ovkit[anomaly]"``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..core.errors import OVKitError
from ..core.results import Masks, Results
from ..pipelines.base import Pipeline


class AnomalyModel(Pipeline):
    """anomalib's ``OpenVINOInferencer``, wrapped to return ovkit results.

    Parameters
    ----------
    model_path:
        Path to the ``model.xml`` anomalib exported (PatchCore, EfficientAD,
        PaDiM, ...).
    metadata:
        Optional ``metadata.json`` written next to the model; it carries the
        normalisation and threshold, without which scores are not comparable.
    threshold:
        Score above which a part is called anomalous. ``None`` uses the
        threshold in the metadata.
    """

    name = "anomaly"
    description = "Score a part as normal or anomalous with an anomalib model."

    def __init__(
        self,
        model_path: str | Path,
        device: str = "AUTO",
        metadata: str | Path | None = None,
        threshold: float | None = None,
    ) -> None:
        super().__init__(device)
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise OVKitError(f"Anomaly model not found: {self.model_path}")
        self.threshold = threshold
        self._inferencer = self._build(metadata)

    def _build(self, metadata: str | Path | None) -> Any:
        try:
            from anomalib.deploy import OpenVINOInferencer
        except ImportError as exc:
            raise ImportError(
                "Anomaly detection needs anomalib.\n"
                '  pip install "ovkit[anomaly]"\n'
                "ovkit ships no anomaly weights: these models are trained on your "
                "own normal samples, so you supply the exported model.xml."
            ) from exc
        kwargs: dict[str, Any] = {"path": self.model_path, "device": self.device}
        if metadata is not None:
            kwargs["metadata"] = Path(metadata)
        return OpenVINOInferencer(**kwargs)

    def run(self, image: np.ndarray, **_: Any) -> Results:
        prediction = self._inferencer.predict(image)
        score = float(getattr(prediction, "pred_score", 0.0))
        threshold = self.threshold
        if threshold is None:
            threshold = float(getattr(prediction, "pred_threshold", None) or 0.5)

        result = Results(image, task=self.name, names={0: "normal", 1: "anomaly"})
        mask = getattr(prediction, "anomaly_map", None)
        if mask is not None:
            result.tensors = {"anomaly_map": np.asarray(mask)}
        segmentation = getattr(prediction, "pred_mask", None)
        if segmentation is not None:
            result.masks = Masks(np.asarray(segmentation, np.uint8)[None])
        verdict = "anomaly" if score >= threshold else "normal"
        result.text = f"{verdict} {score:.2f} (threshold {threshold:.2f})"
        return result
