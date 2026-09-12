#!/usr/bin/env python3
"""Run every capability on a real photo and report what it actually answered.

    python scripts/verify_capabilities.py            # everything
    python scripts/verify_capabilities.py detect scene
    python scripts/verify_capabilities.py --images my_photos/

Or run the **Verify capabilities** workflow on GitHub, which has the network
this container does not.

Why this exists
---------------
ovkit has 19 capabilities and 68 models, and until now nothing ran them end to
end against real weights. Unit tests use fakes: they prove the plumbing, not
that a photo of a street comes back saying "2 people, a car". A capability that
loads, runs, and answers *nothing* passes every test in the suite and is
useless to the person holding the camera — which is exactly what "lots of
models, nothing usable" feels like from outside.

So each case here is judged on its answer, not on the absence of an exception:

``OK``      it ran and said something specific
``EMPTY``   it ran and found nothing — suspicious on a photo chosen to contain
            what it looks for, so this reads as a failure to investigate
``ERROR``   it raised

Capabilities that need motion (``gesture``, ``drowsiness``, ``posture``,
``exercise``, ``track``) cannot be judged on a still, so they are driven with a
short clip built from the photo: that exercises the whole path and catches
crashes, but a ``track`` that finds nothing to track in a still repeated ten
times is not evidence of a bug. Those are marked ``PATH``.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
import urllib.request
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# --- sample photos ---------------------------------------------------------
# Permissively licensed, from the OpenVINO project's own sample data (the
# notebooks repository is Apache-2.0). A URL that has moved shows up as a
# skipped case rather than a silent pass.
NOTEBOOK_DATA = (
    "https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image"
)
IMAGES: dict[str, list[str]] = {
    # key       candidate URLs, tried in order
    #
    # `detect` reported "2x person, airplane, tie" on intel_rnb.jpg and nothing
    # of the sort on coco.jpg, so the people-finders are judged on the former.
    # Judging them on a photo with no people in it produced a page of EMPTY
    # that said nothing about ovkit.
    "street": [f"{NOTEBOOK_DATA}/intel_rnb.jpg", f"{NOTEBOOK_DATA}/coco.jpg"],
    "people": [f"{NOTEBOOK_DATA}/intel_rnb.jpg", f"{NOTEBOOK_DATA}/coco.jpg"],
    "face": [f"{NOTEBOOK_DATA}/coco_hollywood.jpg", f"{NOTEBOOK_DATA}/intel_rnb.jpg"],
    "text": [f"{NOTEBOOK_DATA}/intel_rnb.jpg"],
}


#: ovkit's own vocabulary for "I found nothing". A capability that says one of
#: these ran fine and answered nothing, which on a photo chosen to contain what
#: it looks for is the failure this harness exists to surface — the first run
#: scored every one of them as a pass, which made the report useless.
NOTHING = (
    "nothing found",
    "nothing to anonymise",
    "no face",
    "nobody found",
    "no person",
    "no vehicle",
    "no match",
    "0 instance",
    "cannot tell",
    "empty-looking",
    "못 찾",
    "안 보여",
    "없어요",
    "없습니다",
)


@dataclass
class Case:
    """One capability, the picture it should be judged on, and how to drive it."""

    name: str
    image: str = "street"
    kind: str = "photo"  # photo | clip | audio | setup
    note: str = ""
    kwargs: dict = field(default_factory=dict)
    #: What to print when the same capability appears twice, configured differently.
    label: str = ""

    @property
    def title(self) -> str:
        return self.label or self.name


CASES: list[Case] = [
    # --- single models, the ones a capability name points at ---------------
    Case("detect", "street", note="사람·차 같은 COCO 80종"),
    Case("classify", "street"),
    Case("segment", "street"),
    Case("pose", "people"),
    Case("depth", "street"),
    Case("remove_background", "people"),
    Case("face_detection", "face"),
    Case("text_detection", "text"),
    # --- composed capabilities --------------------------------------------
    Case("scene", "street"),
    Case("face_analyze", "face"),
    Case("person_analyze", "people"),
    Case("vehicle_analyze", "street"),
    Case("read_text", "text", note="표시 언어에 맞는 인식기"),
    Case(
        "read_text",
        "text",
        note="라틴 강제 — 한글 인식기와 비교용",
        label="read_text(latin)",
        kwargs={"recognizer": "text_recognition"},
    ),
    Case("read_plate", "street"),
    Case("count", "street"),
    Case("anonymize", "face"),
    Case("gaze", "face"),
    Case("attention", "face"),
    Case("face_match", "face", kind="setup", note="갤러리에 넣고 다시 물어봄"),
    # --- need motion: the path runs, the answer is not evidence ------------
    Case("track", "street", kind="clip"),
    Case("gesture", "people", kind="clip"),
    Case("drowsiness", "face", kind="clip"),
    Case("posture", "people", kind="clip"),
    Case("exercise", "people", kind="clip"),
    # --- audio -------------------------------------------------------------
    Case("sound_classification", kind="audio"),
    Case("noise_suppression", kind="audio"),
    # --- need something built first: a gallery, a roster, example folders ---
    Case("teach", "people", kind="setup"),
    Case("attendance", "face", kind="setup"),
    # Not a capability: the experiment that separates "the model finds nothing"
    # from "the pipeline never asked it".
    Case("probe", "face", kind="setup", note="진단용 — 왜 파이프라인만 못 찾는가"),
    # `anomaly` needs an anomalib checkpoint and `pip install ovkit[anomaly]`;
    # there is nothing sensible to point it at here, so it is left out rather
    # than reported as passing.
]


def fetch_images(dest: Path, source: Path | None) -> dict[str, Path]:
    """Local photos if given, otherwise the sample set. Missing ones are skipped."""
    if source:
        files = sorted(p for p in source.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not files:
            raise SystemExit(f"no images in {source}")
        return {key: files[i % len(files)] for i, key in enumerate(IMAGES)}

    dest.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    for key, urls in IMAGES.items():
        for url in urls:
            path = dest / f"{key}{Path(url).suffix}"
            try:
                with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
                    path.write_bytes(response.read())
                out[key] = path
                print(f"  {key:8s} <- {url}")
                break
            except Exception as exc:  # noqa: BLE001 - any failure means "try the next"
                print(f"  {key:8s} !! {url} ({type(exc).__name__})")
    return out


def make_clip(image: Path, dest: Path, frames: int = 12) -> Path:
    """A short video of one still — enough to drive the over-time capabilities."""
    import cv2

    frame = cv2.imread(str(image))
    h, w = frame.shape[:2]
    writer = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h))
    for _ in range(frames):
        writer.write(frame)
    writer.release()
    return dest


def make_wav(dest: Path, seconds: float = 2.0, rate: int = 16000) -> Path:
    """A tone under noise — not a real recording, but a real audio path."""
    t = np.linspace(0, seconds, int(rate * seconds), endpoint=False)
    signal = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.05 * np.random.default_rng(0).normal(size=t.size)
    with wave.open(str(dest), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes((np.clip(signal, -1, 1) * 32767).astype("<i2").tobytes())
    return dest


def run_teach(images: dict[str, Path], work: Path) -> tuple[str, str, float]:
    """Learn two classes from two photos, then ask about one of them."""
    import shutil

    from ovkit import Model

    started = time.perf_counter()
    try:
        # Two classes built from the same photo are not two classes. Each key
        # is written to its own file, so comparing paths missed it — compare
        # the bytes.
        import hashlib

        keys: list[str] = []
        digests: set[str] = set()
        for key, path in images.items():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest not in digests:
                digests.add(digest)
                keys.append(key)
        if len(keys) < 2:
            return "SKIP", "서로 다른 사진이 두 장 필요합니다", 0.0
        root = work / "teach"
        shutil.rmtree(root, ignore_errors=True)
        for key in keys[:2]:
            folder = root / key
            folder.mkdir(parents=True)
            for i in range(3):
                shutil.copy(images[key], folder / f"{i}{images[key].suffix}")

        ai = Model("teach")
        for key in keys[:2]:
            ai.learn(key, str(root / key))
        name, score = ai.guess(str(images[keys[0]]))
        elapsed = (time.perf_counter() - started) * 1000
        if name != keys[0]:
            return "EMPTY", f"'{keys[0]}'를 '{name}'로 맞혔습니다 ({score:.2f})", elapsed
        return "OK", f"{name} {score:.2f} (2개 분류, 예시 3장씩)", elapsed
    except Exception as exc:  # noqa: BLE001
        elapsed = (time.perf_counter() - started) * 1000
        return "ERROR", f"{type(exc).__name__}: {str(exc)[:150]}", elapsed


def run_attendance(images: dict[str, Path], work: Path) -> tuple[str, str, float]:
    """Build a one-name roster from the face photo, then take the register."""
    import shutil

    from ovkit import Model

    started = time.perf_counter()
    try:
        roster = work / "roster"
        shutil.rmtree(roster, ignore_errors=True)
        roster.mkdir(parents=True)
        shutil.copy(images["face"], roster / f"학생1{images['face'].suffix}")

        result = Model("attendance", roster=str(roster))(str(images["face"]))
        result = result[-1] if isinstance(result, (list, tuple)) and result else result
        elapsed = (time.perf_counter() - started) * 1000
        said = str(result).replace("\n", " ").strip()
        return ("OK" if said else "EMPTY", said or "(빈 문자열)", elapsed)
    except Exception as exc:  # noqa: BLE001
        elapsed = (time.perf_counter() - started) * 1000
        return "ERROR", f"{type(exc).__name__}: {str(exc)[:150]}", elapsed


def run_probe(images: dict[str, Path], work: Path) -> tuple[str, str, float]:
    """Why does a capability find nothing where its own detector finds something?

    ``Model("face_detection")`` sees a face; ``Model("face_analyze")``, which
    runs that same detector at the same confidence, reports "no face found".
    Same for eight text regions that ``read_text`` reads none of. Guessing has
    been wrong twice, so this asks the three questions that separate the
    possible causes, in one process, and prints the answers.
    """
    import cv2

    from ovkit import Model
    from ovkit.pipelines.base import DEFAULT_CONF

    started = time.perf_counter()
    lines: list[str] = []
    try:
        face = cv2.imread(str(images["face"]))
        text = cv2.imread(str(images["text"]))

        # 1. the detector, called directly, on the array the pipeline would use
        direct = Model("face_detection")(face, conf=DEFAULT_CONF)
        n_direct = len(direct[0].boxes or []) if direct else 0
        lines.append(f"face_detection(ndarray)={n_direct}")

        # 2. the same detector, obtained the way the pipeline obtains it
        pipe = Model("face_analyze")
        via_pipe = pipe.model("face_detection")(face, conf=DEFAULT_CONF)
        n_pipe = len(via_pipe[0].boxes or []) if via_pipe else 0
        lines.append(f"pipeline's detector={n_pipe}")

        # 3. and what the pipeline itself makes of the same array
        n_analyze = len(pipe.run(face).boxes or [])
        lines.append(f"face_analyze(ndarray)={n_analyze}")

        # text: the boxes and the crops they produce
        found = Model("text_detection")(text, conf=DEFAULT_CONF)
        if found:
            boxes = found[0].boxes
            lines.append(f"text boxes={len(boxes or [])} of {text.shape[1]}x{text.shape[0]}")
            for i in range(min(3, len(boxes or []))):
                xyxy = [int(v) for v in boxes.xyxy[i]]
                crop = found[0].crop(i)
                lines.append(f"  box{i}={xyxy} crop={crop.shape if crop.size else 'EMPTY'}")
            reader = Model("text_recognition")
            first = found[0].crop(0)
            if first.size:
                out = reader(first)
                lines.append(f"  recognised={(out[0].text if out else None)!r}")
        return "OK", " · ".join(lines), (time.perf_counter() - started) * 1000
    except Exception as exc:  # noqa: BLE001
        detail = f"{type(exc).__name__}: {str(exc)[:120]}"
        return "ERROR", " · ".join([*lines, detail]), (time.perf_counter() - started) * 1000


def run_face_match(images: dict[str, Path], work: Path) -> tuple[str, str, float]:
    """Add the face to the gallery, then ask who it is.

    Asking an empty gallery produces "no match", which is the right answer to
    the wrong question — it tested nothing.
    """
    from ovkit import Model

    started = time.perf_counter()
    try:
        ids = Model("face_match")
        ids.add("학생1", str(images["face"]))
        name, score = ids.who(str(images["face"])) or (None, 0.0)
        elapsed = (time.perf_counter() - started) * 1000
        if name is None:
            return "EMPTY", "갤러리에 넣은 바로 그 얼굴을 못 알아봤습니다", elapsed
        return "OK", f"{name} {score:.2f} (같은 사진을 넣고 물어봄)", elapsed
    except Exception as exc:  # noqa: BLE001
        return (
            "ERROR",
            f"{type(exc).__name__}: {str(exc)[:150]}",
            (time.perf_counter() - started) * 1000,
        )


SETUP = {
    "teach": run_teach,
    "attendance": run_attendance,
    "face_match": run_face_match,
    "probe": run_probe,
}


def run(case: Case, source: Path) -> tuple[str, str, float]:
    """Return (status, what it said, milliseconds)."""
    from ovkit import Model

    started = time.perf_counter()
    try:
        result = Model(case.name, str(source), **case.kwargs)
        if not isinstance(result, (list, tuple)) and hasattr(result, "__iter__"):
            result = list(result)  # a stream (clip / folder)
        if isinstance(result, (list, tuple)):
            result = result[-1] if result else None
        elapsed = (time.perf_counter() - started) * 1000
        if result is None:
            return "EMPTY", "결과 없음", elapsed
        said = str(result).replace("\n", " ").strip()
        if not said:
            return "EMPTY", "(빈 문자열)", elapsed
        lowered = said.lower()
        if any(phrase in lowered for phrase in NOTHING):
            return "EMPTY", said, elapsed
        # Scores say whether a threshold is the reason something was missed.
        found = getattr(result, "found", None) or []
        if found:
            best = max(float(row.get("score") or 0) for row in found)
            if best:
                said = f"{said}   [top {best:.2f}]"
        return "OK", said, elapsed
    except Exception as exc:  # noqa: BLE001 - the report is the point
        elapsed = (time.perf_counter() - started) * 1000
        detail = str(exc).replace("\n", " ")[:150] or type(exc).__name__
        return "ERROR", f"{type(exc).__name__}: {detail}", elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("only", nargs="*", help="just these capabilities")
    parser.add_argument("--images", default="", help="a folder of your own photos")
    parser.add_argument("--work", default=".verify", help="where samples are written")
    args = parser.parse_args()

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)

    print("samples:")
    images = fetch_images(work, Path(args.images) if args.images else None)
    if not images:
        print("no sample images could be fetched — nothing to verify.", file=sys.stderr)
        return 2

    clips: dict[str, Path] = {}
    wav = make_wav(work / "tone.wav")

    cases = [c for c in CASES if not args.only or c.name in args.only]
    print(f"\n{len(cases)} case(s)\n")
    header = f"{'capability':22s} {'photo':7s} {'status':7s} {'ms':>7s}  answer"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    counts = {"OK": 0, "EMPTY": 0, "ERROR": 0, "SKIP": 0, "PATH": 0}
    failures: list[tuple[str, str]] = []

    for case in cases:
        if case.kind == "setup":
            if case.image not in images:
                counts["SKIP"] += 1
                print(f"{case.title:22s} {case.image:7s} {'SKIP':7s} {'-':>7s}  샘플 사진 없음")
                continue
            status, said, ms = SETUP[case.name](images, work)
            counts[status] = counts.get(status, 0) + 1
            if status == "ERROR":
                failures.append((case.name, said))
            print(
                f"{case.title:22s} {case.image:7s} {status:7s} {ms:7.0f}  {said[:300]}",
                flush=True,
            )
            continue

        if case.kind == "audio":
            source: Path | None = wav
        elif case.image not in images:
            source = None
        elif case.kind == "clip":
            if case.image not in clips:
                clips[case.image] = make_clip(images[case.image], work / f"{case.image}.mp4")
            source = clips[case.image]
        else:
            source = images[case.image]

        print(f"{case.title:22s} {'...':7s} running", flush=True)
        if source is None:
            counts["SKIP"] += 1
            print(f"{case.title:22s} {'SKIP':7s} {'-':>7s}  샘플 사진 없음 ({case.image})")
            continue

        status, said, ms = run(case, source)
        # A still repeated is not motion, so "found nothing" proves nothing here.
        if case.kind == "clip" and status == "EMPTY":
            status = "PATH"
        counts[status] = counts.get(status, 0) + 1
        if status == "ERROR":
            failures.append((case.name, said))
        note = f"   ({case.note})" if case.note else ""
        source_key = "audio" if case.kind == "audio" else case.image
        print(
            f"{case.title:22s} {source_key:7s} {status:7s} {ms:7.0f}  {said[:140]}{note}",
            flush=True,
        )

    print(
        "\n" + "  ".join(f"{k} {v}" for k, v in counts.items() if v) + f"   /  {len(cases)} cases"
    )

    if failures:
        print("\nraised:")
        for name, detail in failures:
            print(f"  {name}: {detail}")
    if counts.get("EMPTY"):
        print(
            "\nEMPTY은 실패로 봅니다 — 그 사진에 있을 법한 것을 못 찾았다는 뜻입니다.\n"
            "샘플이 그 기능에 안 맞는 사진이면 IMAGES/CASES를 고치고, 아니면 기능을 고쳐야 합니다."
        )
    return 1 if failures or counts.get("EMPTY") else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n중단", file=sys.stderr)
        raise SystemExit(130) from None
    except Exception:  # noqa: BLE001 - a harness that dies must say why
        traceback.print_exc()
        raise SystemExit(2) from None
