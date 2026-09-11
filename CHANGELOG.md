# Changelog

## v0.3.0 (2026-08-24)

**ovkit stopped relaying models and started answering questions.** `Model` now
takes a capability name and an input in one call, in Korean or English, and
hands back one readable result.

### Added
- **`Model("기능", 입력)`** — one call, shaped to the input: a photo answers
  with one `Results` (no list indexing for a beginner), a folder with a list, a
  webcam index / video / `"mic"` with a lazy stream.
- **Korean names in and out** — 30+ capability aliases (`얼굴분석`, `졸음감지`,
  `받아쓰기`, …); `Results.found` gives `{name, name_en, score, box, pos}` where
  `name` follows `$OVKIT_LANG` and `name_en` is the stable hyphenated key code
  compares against. ~200 class names translated.
- **`elapsed_ms` and `device` on every result** — with `AUTO` resolved to the
  device that actually ran, because "the NPU did it in 6 ms" is the lesson.
- **`teach`** — a Teachable Machine in five lines: learn from example photos,
  `guess`, `score`, `save`. Five modes (photo / face / hand / upper / body),
  embedding + cosine k-NN, no GPU and no training loop.
- **Classroom four** — `count`, `posture` (neck angle over time), `exercise`
  (squat / push-up reps by joint-angle hysteresis), `attendance` (roster
  matching to a CSV roll).
- **RT-DETR training** — `RTDETR(...).train()/val()/export()` on YOLO-format
  data, Apache-2.0 end to end, behind the `[train]` extra. The exported IR
  drops into `Model(...)` and its `labels.txt` names your classes.
- Capabilities `read_plate`, `attention`, `anonymize`, `gesture`, `scene`,
  `drowsiness`, `gaze`, `track`, `face_match`, `read_text`, `person_analyze`,
  `vehicle_analyze`, plus `anomaly` for your own anomalib model.
- Audio in: `Model("sound_classification")("clip.wav")` reads, resamples,
  frames and decodes; a denoiser's result saves straight back to `.wav`.
- `ovkit gui` (desktop window), `ovkit capabilities`, `ovkit train/val/export`.
- `ovkit.genai.generate / describe / transcribe` — one-call LLM, VLM and
  Whisper that build the tensors the pipeline wants.

### Fixed
- **A capability could be built out of itself** — `Model("gaze")` returned the
  pipeline, and the pipeline then asked for `Model("gaze")` and got itself
  (`'GazeEstimator' object has no attribute 'inputs'`). Sub-models now go
  through `Model.network()`, and two tests keep the trap shut.
- **Contradicting summaries** — a result that had already said "no face found"
  went on to report "2x face". `text` is now the whole answer when it is set.
- **Overlapping overlays** — `plot()` drew up to three captions at once at a
  fixed font size with no wrapping. One wrapped caption on a dark band now.
- **`class_1` / `#3` / `attr_5`** — class tables transcribed from each model's
  documented interface, wired per model in `manifests/labels.yaml`.
- **A generated manifest clobbered hand-written metadata** — `omz.yaml` is
  regenerated and carries only source info, which silently dropped
  `rtdetr_r50`'s COCO names and DETR decode format. Manifests merge key-wise.
- The multi-input message crashed instead of explaining (it interpolated a
  `Model.name` that does not exist).

### Changed
- `ovkit.solutions` and `ovkit.face` (re-export shims) removed; `AnomalyModel`
  moved into `ovkit.pipelines` and registered as `Model("anomaly")`.

## v0.1.2 (2026-08-23)

Every registered model is now verified end-to-end — download, compile, **and
real inference** — 33/33 on Intel® Core™ Ultra hardware.

### Fixed
- **NHWC (channels-last) model inputs** — TF-converted OMZ models
  (`image_retrieval_0001`, `text_detection_0004`,
  `vehicle_license_plate_detection_barrier_0106`) failed at inference; the
  backend now detects channels-last inputs, reads the real spatial size, and
  transposes automatically.
- **PixelLink-style text detectors** — `text_detection_0004`'s
  segm+link logits now decode into scored text boxes.
- `scripts/benchmark.py` survives native device-compiler crashes (each
  model × device cell runs in an isolated subprocess; NPU + dynamic-shape IR
  aborts no longer kill the run).

### Added
- **Web tester redesigned** (`examples/web_app.py`): branded two-tab UI —
  single-model runner (webcam start/stop, image/audio/text) and a
  **full-sweep tab** that tests every registered model one at a time with a
  live progress table (SSE).
- Real CPU/GPU/NPU benchmark table in the README (Core Ultra measurements).
- Windows CPU marketing name in benchmark output.

## v0.1.1 (2026-08-23)

- PyPI project page fixed (absolute logo/link URLs).
- pip-first install docs; PyPI badge.

## v0.1.0 (2026-08-23)

Initial release: one `Model` class over OpenVINO with typed `Results`,
33 representative models (+ capability aliases) served from the HF mirror,
GenAI (LLM/STT), INT8 quantization, `ovkit` CLI, bilingual docs.
