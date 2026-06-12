#!/usr/bin/env python3
"""One-shot model test from the command line.

    python examples/predict.py <model> <image> [--conf 0.25] [--save out.jpg]

Prints a summary (detections / top-5 / mask classes / keypoints / raw tensors)
and saves an annotated image for vision tasks.
"""

from __future__ import annotations

import argparse

import numpy as np

from ovkit import Model


def main() -> None:
    ap = argparse.ArgumentParser(description="Run an ovkit model on one input.")
    ap.add_argument("model", help="registered model name, or path to .xml/.onnx")
    ap.add_argument("source", help="image path (or .npy/.wav for raw models)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="AUTO")
    ap.add_argument("--save", default="out.jpg", help="annotated output path (vision tasks)")
    args = ap.parse_args()

    model = Model(args.model, device=args.device)
    res = model(args.source, conf=args.conf)

    if isinstance(res, dict):  # non-image model -> raw outputs
        print("raw outputs:")
        for name, arr in res.items():
            arr = np.asarray(arr)
            print(f"  {name}: {tuple(arr.shape)} {arr.dtype}")
        return

    r = res[0]
    print(f"task = {model.task}")
    if r.boxes is not None:
        print(f"{len(r.boxes)} detections")
        for x1, y1, x2, y2, c, cl in r.boxes.data[:50]:
            print(f"  {r.name_for(int(cl))} {c:.2f} [{int(x1)},{int(y1)},{int(x2)},{int(y2)}]")
    if r.probs is not None:
        print("top-5:")
        for i in r.probs.top5:
            print(f"  {r.name_for(int(i))}: {r.probs.data[int(i)]:.3f}")
    if r.masks is not None:
        print(f"mask {tuple(r.masks.data.shape)}, classes {np.unique(r.masks.data).tolist()[:25]}")
    if r.keypoints is not None:
        print(f"{len(r.keypoints)} instance(s), {r.keypoints.data.shape[1]} keypoints")
    if r.tensors is not None:
        print("raw outputs:")
        for name, arr in r.tensors.items():
            print(f"  {name}: {tuple(np.asarray(arr).shape)}")

    if r.boxes is not None or r.masks is not None or r.keypoints is not None or r.probs is not None:
        r.save(args.save)
        print(f"saved -> {args.save}")


if __name__ == "__main__":
    main()
