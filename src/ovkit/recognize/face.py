"""Face-analysis adapter — typed decode for the OMZ face model family.

The ``face`` task covers several *different* output kinds (a detector, an
age+gender head, an emotion head, head pose angles, landmark regressors, and
re-identification embeddings). This adapter inspects the output signature and
decodes each into a human-readable :class:`~ovkit.Results` instead of dumping
raw tensors:

- SSD detection ``[1,1,N,7]``       -> ``Results.boxes``
- age (scalar) + gender ``[1,2]``   -> ``Results.text`` ("age 31 · male 98%")
- emotion probabilities ``[1,5]``   -> ``Results.probs`` + top emotion text
- yaw/pitch/roll scalar angles      -> ``Results.text``
- landmark coords ``[1,2K]``        -> ``Results.keypoints`` (pixels)
- embeddings ``[1,D]`` (D >= 64)    -> descriptive text + raw tensor
- anything else                     -> raw tensors (generic fallback)

Output orders (gender = [female, male]; emotions = neutral/happy/sad/
surprise/anger) follow the documented OMZ model interfaces.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.backend import Backend
from ..core.results import Boxes, Keypoints, Probs, Results
from .base import BaseAdapter

_GENDERS = {0: "female", 1: "male"}
_EMOTIONS = {0: "neutral", 1: "happy", 2: "sad", 3: "surprise", 4: "anger"}
#: Head-pose output name -> human label (OMZ: angle_y=yaw, angle_p=pitch, angle_r=roll).
_ANGLES = {"angle_y_fc": "yaw", "angle_p_fc": "pitch", "angle_r_fc": "roll"}
#: Landmark regressors emit [1, 2K] normalized (x, y) pairs; K in {5, 35, 98}.
_LANDMARK_SIZES = frozenset({10, 70, 196})


class FaceAdapter(BaseAdapter):
    """Adapter for face-analysis models (detection, attributes, landmarks)."""

    task = "face"

    def run(self, backend: Backend, image: np.ndarray, conf: float = 0.25, **_: Any) -> Results:
        size = self.model_input_hw(backend)
        rgb = bool(self.pre.get("rgb", False))  # OMZ face models: raw BGR
        feed = self.preprocess(image, size, rgb=rgb, scale=self.pre.get("scale", 1.0))
        outputs = backend.infer(feed)
        return self._decode(image, outputs, conf)

    # -- signature-based decode ----------------------------------------------

    def _decode(self, image: np.ndarray, outputs: dict[str, Any], conf: float) -> Results:
        h, w = image.shape[:2]
        arrs = {n: np.asarray(a) for n, a in outputs.items()}

        # SSD-style detector: any [.., N, 7] output.
        for arr in arrs.values():
            if arr.ndim >= 3 and arr.shape[-1] == 7:
                det = arr.reshape(-1, 7)
                keep = det[:, 2] >= conf
                det = det[keep]
                xyxy = det[:, 3:7] * np.array([w, h, w, h], dtype=np.float32)
                xyxy[:, 0::2] = xyxy[:, 0::2].clip(0, w)
                xyxy[:, 1::2] = xyxy[:, 1::2].clip(0, h)
                data = np.concatenate(
                    [xyxy, det[:, 2:3], np.zeros_like(det[:, 1:2])], axis=1
                ).astype(np.float32)
                return Results(image, task="detect", names={0: "face"}, boxes=Boxes(data))

        # Age + gender: a scalar "age" output next to a 2-class "prob" output.
        age_name = next((n for n in arrs if "age" in n.lower() and arrs[n].size == 1), None)
        gen_name = next((n for n in arrs if n != age_name and arrs[n].size == 2), None)
        if age_name and gen_name:
            age = float(arrs[age_name].reshape(-1)[0]) * 100.0
            g = arrs[gen_name].reshape(-1).astype(np.float32)
            gi = int(np.argmax(g))
            r = Results(image, task=self.task, names=_GENDERS, probs=Probs(g))
            r.text = f"age {age:.0f} · {_GENDERS[gi]} {g[gi] * 100:.0f}%"
            return r

        # Head pose: three scalar angle outputs.
        angles = {n: a for n, a in arrs.items() if a.size == 1 and "angle" in n.lower()}
        if len(angles) == 3:
            parts = []
            for key, label in _ANGLES.items():
                match = next((n for n in angles if n.startswith(key)), None)
                if match is not None:
                    parts.append(f"{label} {float(angles[match].reshape(-1)[0]):+.1f}°")
            r = Results(image, task=self.task, names=self.names, tensors=outputs)
            r.text = " · ".join(parts)
            return r

        if len(arrs) == 1:
            flat = next(iter(arrs.values())).reshape(-1).astype(np.float32)

            # Emotion head: 5 probabilities.
            if flat.size == 5:
                ei = int(np.argmax(flat))
                r = Results(image, task=self.task, names=_EMOTIONS, probs=Probs(flat))
                r.text = f"{_EMOTIONS[ei]} {flat[ei] * 100:.0f}%"
                return r

            # Landmark regressor: 2K normalized coords (not a prob distribution).
            if flat.size in _LANDMARK_SIZES and np.all(np.isfinite(flat)):
                if float(flat.min()) >= -0.5 and float(flat.max()) <= 1.5:
                    pts = flat.reshape(-1, 2)
                    kpts = np.stack(
                        [pts[:, 0] * w, pts[:, 1] * h, np.ones(len(pts), np.float32)], axis=1
                    )
                    return Results(
                        image, task=self.task, names=self.names, keypoints=Keypoints(kpts[None])
                    )

            # Two-class head with no age companion (e.g. anti-spoof: [real, spoof]).
            if flat.size == 2:
                names = {0: "real", 1: "spoof"}
                i = int(np.argmax(flat))
                r = Results(image, task=self.task, names=names, probs=Probs(flat))
                r.text = f"{names[i]} {flat[i] * 100:.0f}%"
                return r

            # Re-identification embedding.
            if flat.size >= 64:
                r = Results(image, task=self.task, names=self.names, tensors=outputs)
                r.text = f"{flat.size}-d face embedding (use for matching)"
                return r

        return Results(image, task=self.task, names=self.names, tensors=outputs)
