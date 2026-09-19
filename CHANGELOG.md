# Changelog

## Unreleased

### Changed
- **`ovkit run` streams video and opens a camera.** It used to go through
  `predict` and read a whole clip into a list before printing a single line,
  and `ovkit run detect 0` looked for a file called `0`. Now it follows the
  same rule as `Model(name, source)`: a photo prints once and saves
  `<name>_out.jpg`; a folder prints one line per file; a video or a camera
  index prints every frame as it arrives, in a window when there is one,
  `q` or Ctrl-C to stop, `--save` for the last frame.
- The cookbook (EN/KO) still said `model(x)` returns a `list[Results]` — it has
  not since 0.5.0. It also taught `cv2.imwrite(...)` for saving an overlay,
  which is the exact call that lost a Korean filename on Windows in 0.4.0;
  `r.save(...)` now.
- `CONTRIBUTING.md` states the `.predict()` rule for pipeline authors that the
  0.5.0 guard test enforces.

## v0.5.0 (2026-09-13)

**Keeping a model to reuse it no longer changes the answer.** `Model(이름, 입력)`
shaped its answer to the input; `m(입력)` did not, because it was a plain alias
for `predict` and always returned a list. So the reusable form — the one a
lesson or a loop actually uses — was the awkward one.

```python
m = Model("장면설명")
m("교실.jpg")          # 0.4.1: list of one, needed [0].  Now: one Results.
m("우리반/")           # a list, as before
for r in m(0): ...     # 0.4.1: tried to collect a webcam. Now: a stream.
```

### Changed
- **`model(x)` and `pipeline(x)` now shape the answer to the input**, by the
  same rule `Model(name, x)` has always used: one image, sound file or sentence
  gives one `Results`; a folder or a list gives a list; a camera index, a video
  or `"mic"` gives a lazy stream. **This is a breaking change** — code that
  wrote `m("photo.jpg")[0]` should drop the `[0]`.
- **`m(0)` no longer hangs.** It used to ask `predict` for a list and try to
  read a webcam to its end.
- **`predict()` is unchanged** and is now the documented uniform form: always a
  list, or a generator with `stream=True`. Passing `stream=` to `m(x)` hands
  the call straight to it. Library code should call `predict`.
- **ovkit's own pipelines call `predict`.** Twenty sub-model calls inside `src`
  used the short form and expected a list; all of them say `.predict(...)` now,
  and a test fails if a new one does not. Same lesson as `imread`: a rule only
  holds while the call sites follow it.

How many things were *found* is untouched — it was never this layer. Ten people
in one photo is one `Results` with ten entries in `r.found`, both before and
after.

## v0.4.1 (2026-09-13)

**A Korean filename broke two capabilities on Windows.** Both shipped in 0.4.0.
Found by running the test suite on the machine ovkit is actually for — a Korean
Windows laptop — which is also why CI now has a Windows job.

### Fixed
- **`Model("출석체크")` could not read its own roster.** A folder of students'
  names is a folder of Korean filenames, and `cv2.imread` hands the path to the
  C++ runtime, which reads it in the system code page: `철수.png` arrived as
  `泥좎닔.png` and "did not exist". `imread`/`imwrite` now read and write the
  bytes in Python and let OpenCV do only the decoding.
- **`collect("가위", 30)` lost the photos it took.** Same cause, in
  `pipelines/teach.py` — the caller names the folder, and the docs' own example
  names it in Korean. The frames were written somewhere nobody could find again.
- **Three tests could not even run on a Korean Windows machine.**
  `Path.read_text()` with no `encoding` opens in the locale encoding, which is
  cp949 there, and ovkit's own sources are full of em dashes. Every
  `read_text`/`write_text` in the repository is now explicit.

### Changed
- **CI runs on Windows too** (`windows-latest`, 3.12, alongside Ubuntu
  3.10–3.12). Neither bug above can reproduce on Linux, so an Ubuntu-only
  matrix was never going to catch them.
- A test now fails if anything outside `image/ops.py` calls `cv2.imread` or
  `cv2.imwrite`. Fixing `ops.py` had not been enough: nineteen call sites went
  around it, one of them in `src`.
- `tests/assets/sample.png` — the image the file-I/O tests read, so they decode
  bytes this repository actually ships. Four flat quadrants in known BGR values,
  which catches a swapped-channel or half-read decode that a shape check misses.

## v0.4.0 (2026-09-12)

**ovkit learned to speak, and learned to read Hangul.** Both go through the
same `Model(...)` call, and every capability was run against real weights
before this release — which is how most of the fixes below were found.

### Added
- **`Model("읽어주기", "안녕하세요")`** — text to speech in **31 languages**,
  Korean among them. Four networks chained (duration → text encoder → flow
  matching → vocoder), 44.1 kHz, ten voices, under a second per sentence on a
  laptop CPU. `r.save("인사.wav")` writes the sound, `r.save("인사.png")` the
  waveform. Supertonic 3, OpenRAIL-M, mirrored with its licence.
- **Korean OCR** — `read_text` picks the recogniser that matches the display
  language: PP-OCRv3 Korean for `ko`, the Latin one for `en`. Force either with
  `recognizer=`.
- **`ovkit pull`** — convert any Hugging Face model once and run it with plain
  ovkit afterwards.
- **RT-DETR ladder** — `rtdetr_r18` / `r34` / `r50` / `r101`, r18 by default.
- `depth` and `remove_background`, mirrored out of their AGPL upstream.
- Seven models that were sitting unreferenced in the mirror, now registered
  (98-point facial landmarks, person re-id, two super-resolutions, …).
- **`docs/features.md`** and **`docs/ko/features.md`** — every capability, what
  it answers with, and what it is made of.
- `ovkit gui` speaks Korean, and has a text box for 읽어주기.
- `scripts/check.py` — runs exactly what CI runs, with real exit codes.

### Fixed
- **Every pipeline's sub-models were built with the wrong arguments.**
  `Model.network()` passed `task`/`device`/`precision` positionally into a
  signature whose second parameter is `source`, so every sub-model was created
  with `task="AUTO"`, fell back to the generic adapter, and returned no boxes.
  Ten capabilities answered "nothing found" on pictures full of the thing they
  look for. Keyword arguments, and two regression tests.
- **The Latin text recogniser was off by one letter.** `text-recognition-0014`
  puts its blank first; ovkit's default table put it last, so "building" came
  out `c0v0j0me0joh0`.
- **A capability was stricter than the model it wraps** — `face_detection`
  found a face at 0.47 and `face_analyze`, hardcoded to 0.5, reported none on
  the same photo. One `DEFAULT_CONF`, and the caller's `conf` now reaches the
  sub-pipelines.
- **`attendance` recognised nobody** — it enrolled whole roster photos and
  matched face crops.
- **"Found nothing" and "read nothing" were the same answer.** `read_text` now
  warns when it finds text regions and reads none of them.
- The Korean character table lost its space entry (3687 classes instead of
  3690), so spaces were never emitted.
- `sound_classification` crashed on models with a dynamic sample axis.
- `Results.show()` killed the process on a machine with no display (full
  OpenCV exits through Qt, uncatchable). It checks first, and saves instead.
- The first 60 seconds were silent — a 125 MB download with no output. Now it
  says what it is fetching and how far along it is.

### Changed
- **One mirror.** Everything ovkit serves comes from
  `leeyunjai/ovkit-models` — 278 files, 2.8 GB, every one referenced by a
  manifest. External hosts remain only as fallbacks for when the mirror is
  unreachable.
- **The licence allow-list opened, narrowly.** OpenRAIL-M loads when the entry
  can point at its licence (`license_url`), because that licence requires every
  copy to pass on the same use restrictions. AGPL and CC-BY-NC stay out.
- **`tier: part`** — a network that is a piece of a capability and does nothing
  alone (a vocoder wants a latent) is hidden from every listing a person reads,
  without being filed under `zoo`, which means "superseded".
- `manifests` entries may declare `data:` — the tables and licences a model
  needs beside its weights, so mirror pruning cannot delete them.

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
