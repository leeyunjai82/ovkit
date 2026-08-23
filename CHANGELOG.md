# Changelog

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
