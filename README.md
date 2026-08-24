<div align="center">

<img src="https://raw.githubusercontent.com/leeyunjai82/ovkit/main/docs/_static/logo.svg" width="110" alt="ovkit logo"/>

# ovkit

**OpenVINO inference in 3 lines.** One `Model` class, clean `Results`, 30+
ready models — with `AUTO`/`NPU`/`GPU` devices, async throughput, and INT8.

[![CI](https://github.com/leeyunjai82/ovkit/actions/workflows/ci.yml/badge.svg)](https://github.com/leeyunjai82/ovkit/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-github.io-0a7d8c)](https://leeyunjai82.github.io/ovkit/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://github.com/leeyunjai82/ovkit/blob/main/pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](https://github.com/leeyunjai82/ovkit/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/ovkit)](https://pypi.org/project/ovkit/)
[![Models](https://img.shields.io/badge/🤗%20models-ovkit--models-yellow)](https://huggingface.co/leeyunjai/ovkit-models)

[Docs](https://leeyunjai82.github.io/ovkit/) ·
[한국어 문서](https://leeyunjai82.github.io/ovkit/ko/) ·
[Model catalog](https://leeyunjai82.github.io/ovkit/models.html) ·
[Examples](https://github.com/leeyunjai82/ovkit/tree/main/examples)

</div>

```python
from ovkit import Model

r = Model("detect", "image.jpg")      # download -> convert -> cache -> run
print(r)                              # 2x person, car
r.save("out.jpg")                     # the boxes drawn on the photo
```

One call fits every input — and Korean names work too:

```python
r = Model("얼굴분석", "group.jpg")      # == Model("face_analyze", ...)
for row in r.found:                    # [{'name': '사람', 'name_en': 'person',
    print(row["name"], row["pos"])     #   'score': 0.93, 'box': [...], 'pos': '왼쪽 위'}]

for r in Model("track", 0):            # webcam / video / "mic" -> a stream
    print(r, r.elapsed_ms, r.device)   # 2x person (#1, #4) 14.2 GPU
```

The same call also runs whole **capabilities** — several models chained into one
answer, so you never have to crop, chain and stitch by hand:

```python
r = Model("face_analyze", "group.jpg")
print(r)                # 2 faces: age 31 · male 98% · happy 92%, age 27 · female 95% · neutral 88%
```

Or without writing Python at all:

```bash
ovkit gui                             # a window: pick a capability, point it at your webcam
ovkit run detect image.jpg            # prints results, saves image_out.jpg
ovkit capabilities                    # what Model(name) can answer
```

## Install

```bash
pip install ovkit
```

Extras: `ovkit[quant]` (INT8/NNCF) · `ovkit[genai]` (LLM/STT) · `ovkit[all]`.
Python 3.10+. For development, install from source:

```bash
git clone https://github.com/leeyunjai82/ovkit.git && cd ovkit
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Capabilities

A capability name gives you the answer, not the plumbing. Each one chains a
detector with the models that describe what it found — same `Model(...)` call,
same `Results`, and it takes the same sources (path, ndarray, folder, video,
camera index).

| `Model(...)` | Answers | Chains |
| ------------ | ------- | ------ |
| `scene` | `2 people (1 happy) · a laptop and a cup · floor 47%` | detection + segmentation + faces |
| `face_analyze` | `2 faces: age 31 · male 98% · happy 92%, ...` | face detection + age/gender + emotion (+ head pose, landmarks) |
| `person_analyze` | `3 people: male 0.98 · long pants 0.95 · bag 0.71, ...` | person detection + attributes |
| `vehicle_analyze` | `2 vehicles: type: car (0.98) · color: black (0.83), ...` | vehicle detection + type/colour |
| `read_text` | `'STOP AHEAD'` — every word, in reading order | text detection + text recognition |
| `read_plate` | `2 vehicles: black car — 12GA3456, ...` | plate detection + text recognition + vehicle attributes |
| `track` | `2x person (#1, #4)` — ids stable across frames | detection + IoU association |
| `drowsiness` | `EYES CLOSED 1.4s — drowsy` | face + landmarks + eye state + head pose, **over time** |
| `gesture` | `thumb up 0.94` | sign-language model over a rolling 8-frame clip |
| `gaze` | `1 face: looking right and slightly up` | face detection + landmarks + head pose + gaze |
| `attention` | `1 person looking at: laptop` | gaze + object detection (ray-cast into the boxes) |
| `anonymize` | the picture with every face pixelated | face (and plate) detection + redaction |
| `face_match` | `('yunjai', 0.81)` — who this is | embedding + cosine matching against your gallery |

```python
from ovkit import Model, list_pipelines

list_pipelines()                                    # every capability, described
Model("read_text")("sign.jpg")[0].text              # 'STOP AHEAD'
Model("track")(0)                                   # webcam, ids kept across frames
Model("face_analyze", attributes=("age_gender",))   # configure what runs
```

Aliases: `ocr`, `anpr`, `blur`, `driver`, `describe`, `faces`, `people`, `vehicle`, `tracking`, `reid`.

## Teach your own AI (no GPU)

A Teachable Machine in five lines — an embedding model turns examples into
vectors, new inputs match the nearest ones. Five modes: `photo` (default),
`face` (expressions), `hand` (needs `ovkit[hand]`), `upper`, `body`:

```python
ai = Model("teach")                       # or Model("가르치기")
ai.learn("can", "photos/cans/")           # a folder per thing to recognise
ai.learn("bottle", "photos/bottles/")
print(ai.guess("new_photo.jpg"))          # ('can', 0.93)
print(ai.score("test_photos/"))           # accuracy + what it confuses
ai.save("recycling")                      # -> Documents/ovkit/recycling.json
for r in ai.predict(0, stream=True): ...  # webcam over YOUR classes
```

Korean method names work too: `배우기`, `맞혀봐`, `점수`, `저장`. Collect
examples with the webcam: `from ovkit.pipelines.teach import collect; collect("가위", 30)`.

## Train your own detector

RT-DETR, trainable, Apache-2.0 end to end — an Ultralytics-style workflow with
no AGPL code or weights anywhere (`pip install "ovkit[train]"`):

```python
from ovkit import RTDETR

model = RTDETR("rtdetr-r18")                 # or RTDETR("best.pt") to resume
model.train(data="data.yaml", epochs=100)    # the YOLO-format labels you already have
model.val(data="data.yaml")                  # mAP50 / mAP50-95
model.export(half=True)                      # -> IR + labels.txt

r = model("bus.jpg", conf=0.5)               # ovkit Results: print(r), r.found, r.save()
```

The exported IR drops straight into `Model("path/to/rtdetr-r18.xml")` — the
`labels.txt` written next to it means your classes answer by name. CLI:
`ovkit train --data data.yaml`, `ovkit val`, `ovkit export`.
`ovkit capabilities` prints the list; **`ovkit gui` opens a window** where you can
click through them against your webcam or a picture.

## Supported tasks

Every single-model task below runs end-to-end through the same 3 lines — swap the alias:

| Task | Alias | Output | Example |
| ---- | ----- | ------ | ------- |
| Object detection | `detect`, `face_detection`, `person_detection`, `vehicle_detection`, `text_detection`, `license_plate` | `r.boxes` | [detect.py](https://github.com/leeyunjai82/ovkit/blob/main/examples/detect.py) |
| Classification | `classify`, `person_attributes`, `vehicle_attributes` | `r.probs` / text | [classify.py](https://github.com/leeyunjai82/ovkit/blob/main/examples/classify.py) |
| Segmentation | `segment`, `instance_segmentation` | `r.masks` | [segment.py](https://github.com/leeyunjai82/ovkit/blob/main/examples/segment.py) |
| Pose / landmarks | `pose`, `face_landmarks` | `r.keypoints` | [pose.py](https://github.com/leeyunjai82/ovkit/blob/main/examples/pose.py) |
| Face analysis | `age_gender`, `emotion`, `head_pose`, `face_reid` | `r.text` (e.g. `"age 31 · male 98%"`) | [face_analysis.py](https://github.com/leeyunjai82/ovkit/blob/main/examples/face_analysis.py) |
| OCR | `text_recognition` | `r.text` | [ocr.py](https://github.com/leeyunjai82/ovkit/blob/main/examples/ocr.py) |
| Tracking / matching | `track`, `face_match` | `r.track_ids` · `(label, score)` | [track.py](https://github.com/leeyunjai82/ovkit/blob/main/examples/track.py) / [face_match.py](https://github.com/leeyunjai82/ovkit/blob/main/examples/face_match.py) |
| Super-resolution | `super_resolution` | upscaled image via `r.plot()` | [super_resolution.py](https://github.com/leeyunjai82/ovkit/blob/main/examples/super_resolution.py) |
| LLM / STT (GenAI) | `llm`, `stt` | generated text | [llm.py](https://github.com/leeyunjai82/ovkit/blob/main/examples/llm.py) / [stt.py](https://github.com/leeyunjai82/ovkit/blob/main/examples/stt.py) |
| NLP / audio / time series | `qa`, `translation`, `noise_suppression`, `time_series` | tensors via `model.infer()` | [denoise_audio.py](https://github.com/leeyunjai82/ovkit/blob/main/examples/denoise_audio.py) |

The registry exposes **one well-tested model per capability (35 total)**; the
[HF mirror](https://huggingface.co/leeyunjai/ovkit-models) hosts the full
Apache-2.0 OMZ set (other tiers, `int8`, `sparse` variants) — surfacing a
variant is a one-line edit ([catalog](https://leeyunjai82.github.io/ovkit/models.html)).
`ovkit list` shows everything with descriptions.

## Devices

| Device | How | Notes |
| ------ | --- | ----- |
| **AUTO** (default) | `Model("detect")` | OpenVINO picks the best device |
| **CPU** | `Model("detect", device="CPU")` | works everywhere |
| **GPU** | `device="GPU"` | Intel iGPU / Arc |
| **NPU** | `device="NPU"` | Intel® Core™ Ultra AI accelerator |

Single images run synchronously; `stream=True` uses an `AsyncInferQueue` for
video/webcam throughput. INT8: `model.quantize(calib_images)` (NNCF).

### Benchmarks

```bash
python scripts/benchmark.py        # prints a paste-ready CPU/GPU/NPU table
```

| model | CPU | GPU | NPU |
| --- | --- | --- | --- |
| `rtdetr_r50` | 429.4 ms (2 FPS) | 36.9 ms (27 FPS) | —* |
| `face_detection_0205` | 11.7 ms (85 FPS) | 4.5 ms (224 FPS) | —* |
| `person_detection_0202` | 13.4 ms (75 FPS) | 4.7 ms (211 FPS) | 6.6 ms (151 FPS) |
| `resnet50_binary_0001` | 7.9 ms (126 FPS) | 5.1 ms (195 FPS) | —* |
| `road_segmentation_adas_0001` | 28.6 ms (35 FPS) | 13.0 ms (77 FPS) | 23.8 ms (42 FPS) |
| `human_pose_estimation_0007` | 82.2 ms (12 FPS) | 14.0 ms (71 FPS) | 22.7 ms (44 FPS) |
| `age_gender_recognition_retail_0013` | 0.5 ms (1867 FPS) | 0.5 ms (2098 FPS) | 0.6 ms (1569 FPS) |

*Measured on an Intel® Core™ Ultra (Lunar Lake) laptop — CPU / integrated GPU /
NPU, median of 30 runs, 1280x720 input, OpenVINO 2026.3.*
`—` = model not supported by the NPU compiler (dynamic shapes or unsupported ops).

## Usage

<details open>
<summary><b>Python</b></summary>

```python
from ovkit import Model

model = Model("face_detection")              # alias, name, .xml, or .onnx
results = model("photo.jpg", conf=0.25)      # image / ndarray / folder / video
for r in model.predict(0, stream=True):      # webcam (lazy generator)
    annotated = r.plot()

print(Model("age_gender")("face.jpg")[0].text)   # "age 31 · male 98%"
```

Inputs are **auto-detected**: image path / `ndarray` / folder / video / camera
index → vision pipeline; `.npy` / `.wav` → raw inference. Grayscale models and
all-image multi-input models (super-resolution) are handled automatically. Full
control for any model: `model.infer({name: tensor})` with `model.inputs`.

| `Results` | holds |
| --------- | ----- |
| `r.boxes` | `xyxy`, `xywh`, `conf`, `cls` |
| `r.masks` / `r.keypoints` / `r.probs` | masks · `[x,y,conf]` · `top1`/`top5` |
| `r.text` | decoded text (OCR, face attributes) |
| `r.labels` / `r.track_ids` | per-box label a pipeline wrote · per-box track id |
| `r.tensors` | raw `{name: ndarray}` |
| `r.summary()` | the whole result as one readable line |
| `r.crop(i)` / `r.to_dict()` / `r.to_json()` | cut a box out · plain Python · JSON |
| `r.plot()` / `r.save(path)` | annotated image (or the model's output image) |

</details>

<details>
<summary><b>CLI</b></summary>

```bash
ovkit run detect image.jpg --save out.jpg   # one-shot inference
ovkit run age_gender face.jpg --device NPU
ovkit list                                  # aliases + models with descriptions
ovkit info face_detection                   # source / task / license
ovkit download detect                       # warm the cache
ovkit devices                               # available OpenVINO devices
```

</details>

<details>
<summary><b>GenAI (LLM / speech-to-text)</b></summary>

```python
from ovkit.genai import pipeline

llm = pipeline("llm")                        # tinyllama_chat from the mirror
print(llm.generate("Explain OpenVINO in one sentence.", max_new_tokens=64))

stt = pipeline("stt")                        # whisper_base
print(stt.generate(audio_16k_mono_float32))
```

Needs `pip install "ovkit[genai]"`.

</details>

<details>
<summary><b>Web demo (image / webcam / audio / text)</b></summary>

```bash
pip install -r examples/requirements.txt
python examples/web_app.py                   # http://127.0.0.1:8000
```

Pick any model — the right input (upload / webcam / audio / text) appears
automatically and results render with overlays.

</details>

## Adding a model

Models are data, not code — one manifest entry
(`src/ovkit/manifests/`):

```yaml
my_model:
  src: hf
  repo: leeyunjai/ovkit-models
  filename: detect/my_model/model.xml
  task: detect
  description: Shown by `ovkit list`.
  license: apache-2.0            # must be permissive — enforced at load time
```

Resolution: alias → local path → cache (`~/.cache/ovkit`) → download → convert
→ cache, with atomic writes, `sha256` checks, upstream `fallback`, and
`OVKIT_OFFLINE=1`. Maintainer tooling (mirror build / verify / self-check /
benchmark) lives in [`scripts/`](https://github.com/leeyunjai82/ovkit/tree/main/scripts) — see the
[guide](https://leeyunjai82.github.io/ovkit/guide.html).

## License

ovkit is [Apache-2.0](https://github.com/leeyunjai82/ovkit/blob/main/LICENSE) and **license-clean by design**: only permissive
(Apache/MIT/BSD) models and libraries — no AGPL model stacks, no non-commercial
weights; every manifest entry must declare a permissive license (enforced at
load time).
