#!/usr/bin/env python3
"""Benchmark registered models across OpenVINO devices; print a README table.

Measures single-image latency (median over N runs, after warmup) and FPS per
available device (CPU / GPU / NPU), then prints a ready-to-paste Markdown
table. Run it on your own hardware and paste the output into README.md::

    python scripts/benchmark.py                       # default model set
    python scripts/benchmark.py --models detect pose  # subset (aliases OK)
    python scripts/benchmark.py --devices CPU NPU --runs 50

Models are pulled from the mirror on first use (cached afterwards).
"""

from __future__ import annotations

import argparse
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

#: Good defaults: one light + one heavy model per common vision task.
_DEFAULT_MODELS = [
    "detect",  # rtdetr_r50 (heavier, transformer)
    "face_detection",  # face_detection_0205
    "person_detection",  # person_detection_0202
    "classify",  # resnet50_binary_0001
    "segment",  # road_segmentation_adas_0001
    "pose",  # human_pose_estimation_0007
    "age_gender",  # age_gender_recognition_retail_0013 (tiny)
]


def _cpu_name() -> str:
    if platform.system() == "Windows":
        try:  # marketing name (e.g. "Intel(R) Core(TM) Ultra 7 258V"), not the family/model id
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            return platform.processor() or "unknown CPU"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def bench_one(name: str, device: str, runs: int, warmup: int) -> tuple[float, float] | str:
    """Return (median_ms, fps) for one model on one device, or an error string."""
    from ovkit import Model

    try:
        model = Model(name, device=device)
        img = np.random.randint(0, 255, (720, 1280, 3), np.uint8)
        for _ in range(warmup):
            model(img)
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            model(img)
            times.append((time.perf_counter() - t0) * 1000.0)
        med = statistics.median(times)
        return med, 1000.0 / med
    except Exception as exc:
        return f"{type(exc).__name__}: {str(exc)[:60]}"


def bench_isolated(name: str, device: str, runs: int, warmup: int) -> tuple[float, float] | str:
    """Run one benchmark in a subprocess so a native crash can't kill the run.

    Some device compilers (e.g. the NPU compiler on dynamic-shape IR) abort the
    whole process with a native error that Python cannot catch; isolating each
    (model, device) cell keeps the rest of the table alive.
    """
    import subprocess

    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_bench-one",
        name,
        device,
        str(runs),
        str(warmup),
    ]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        return "timeout (30 min)"
    for line in reversed((p.stdout or "").splitlines()):
        if line.startswith("OVKIT_BENCH_RESULT "):
            _, med, fps = line.split()
            return float(med), float(fps)
        if line.startswith("OVKIT_BENCH_ERROR "):
            return line[len("OVKIT_BENCH_ERROR ") :]
    tail = [ln for ln in (p.stderr or "").splitlines() if ln.strip()]
    reason = tail[-1][:70] if tail else f"crashed (exit {p.returncode})"
    return f"native crash: {reason}"


def main(argv: list[str] | None = None) -> int:
    # Hidden child mode used by bench_isolated().
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["--_bench-one"]:
        name, device, runs, warmup = argv[1], argv[2], int(argv[3]), int(argv[4])
        r = bench_one(name, device, runs, warmup)
        if isinstance(r, tuple):
            print(f"OVKIT_BENCH_RESULT {r[0]:.3f} {r[1]:.3f}")
        else:
            print(f"OVKIT_BENCH_ERROR {r}")
        return 0

    ap = argparse.ArgumentParser(description="Benchmark ovkit models across devices.")
    ap.add_argument("--models", nargs="*", default=_DEFAULT_MODELS, help="names or aliases")
    ap.add_argument("--devices", nargs="*", default=None, help="default: all available")
    ap.add_argument("--runs", type=int, default=30, help="timed runs per model (default 30)")
    ap.add_argument("--warmup", type=int, default=5, help="warmup runs (default 5)")
    args = ap.parse_args(argv)

    from ovkit.core.backend import available_devices
    from ovkit.core.registry import resolve

    devices = args.devices or [d for d in available_devices() if "." not in d]
    print(f"# Hardware: {_cpu_name()}")
    print(f"# Devices : {devices}   ({args.runs} runs, median, 1280x720 input)\n")

    rows: list[list[str]] = []
    for name in args.models:
        entry = resolve(name)
        real = entry.name if entry else name
        row = [f"`{real}`"]
        for dev in devices:
            print(f"benchmarking {real} on {dev}...", flush=True)
            r = bench_isolated(name, dev, args.runs, args.warmup)
            row.append(f"{r[0]:.1f} ms ({r[1]:.0f} FPS)" if isinstance(r, tuple) else "—")
            if not isinstance(r, tuple):
                print(f"  {dev}: {r}")
        rows.append(row)

    print("\nPaste into README.md:\n")
    print("| model | " + " | ".join(devices) + " |")
    print("|" + " --- |" * (len(devices) + 1))
    for row in rows:
        print("| " + " | ".join(row) + " |")
    print(f"\n*Measured on {_cpu_name()} — median of {args.runs} runs, 1280x720 input.*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
