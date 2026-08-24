---
sd_hide_title: true
---

# ovkit

:::{div} ovk-hero
# ovkit

```{div} ovk-tagline
A simple Python inference API for OpenVINO — one import, one `Model` class, a
callable object, and clean `Results`, with `AUTO`/`NPU` devices, async
throughput, and INT8 quantization.
```

```{div} ovk-badges
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-Apache--2.0-green)
![OpenVINO](https://img.shields.io/badge/OpenVINO-IR%20%7C%20genai-0a7d8c)
```
:::

```python
from ovkit import Model

model = Model("rtdetr_r50")            # name -> auto download / convert / cache
for r in model("img.jpg", conf=0.25):  # __call__ == predict
    print(r.boxes.xyxy, r.boxes.conf, r.boxes.cls)
    r.save("out.jpg")
```

::::{grid} 1 2 2 3
:gutter: 3

:::{grid-item-card} 🚀 Get started
:link: usage
:link-type: doc
Install, load a model, run prediction, pick a device.
:::

:::{grid-item-card} 📖 Cookbook
:link: cookbook
:link-type: doc
Copy-paste call examples for every feature.
:::

:::{grid-item-card} 🧩 API reference
:link: api
:link-type: doc
`Model`, `Results`, registry, adapters, image ops.
:::
::::

## Capabilities

A capability name gives you the answer, not the plumbing — one `Model(...)`
call chains a detector with the models that describe what it found.

```python
from ovkit import Model, list_pipelines

list_pipelines()                                   # every capability, described
Model("face_analyze")("group.jpg")[0].summary()    # '2 faces: age 31 · male 98% · happy 92%, ...'
Model("read_text")("sign.jpg")[0].text             # 'STOP AHEAD'
Model("track")(0)                                  # webcam, ids stable across frames
```

| `Model(...)` | Answers | Chains |
| ------------ | ------- | ------ |
| `face_analyze` | ages, genders and emotions of every face | face detection + age/gender + emotion |
| `person_analyze` | what each person wears or carries | person detection + attributes |
| `vehicle_analyze` | type and colour of each vehicle | vehicle detection + attributes |
| `read_text` | every word, in reading order | text detection + recognition |
| `track` | a stable id per object across frames | detection + IoU association |
| `gaze` | where a face is looking | detection + landmarks + head pose + gaze |
| `face_match` | who this is, from your own gallery | embedding + cosine matching |
| `scene` | one sentence about the whole picture | detection + segmentation + faces |
| `read_plate` | number plates, and the car each is on | plate detection + OCR + vehicle attributes |
| `drowsiness` | eyes shut too long, or a nodding head | face + landmarks + eye state + head pose, over time |
| `gesture` | hand gestures from motion | sign-language model over a rolling 8-frame clip |
| `attention` | which object a person is looking at | gaze + object detection |
| `anonymize` | the picture with faces (and plates) removed | detection + redaction |

## What it does

::::{grid} 1 2 2 2
:gutter: 2

:::{grid-item-card} 🎯 Vision tasks
Detection (DETR / SSD / YOLO), classification, segmentation (semantic +
instance), pose, OCR — `model(img)` → typed `Results`.
:::

:::{grid-item-card} ⚙️ Any model
Generic raw-tensor fallback + low-level `model.infer(tensors)` for NLP / audio /
time-series.
:::

:::{grid-item-card} 💬 GenAI
LLM / Whisper (STT) / TTS via `ovkit.genai.pipeline(...)` (openvino-genai).
:::

:::{grid-item-card} 📦 Auto everything
Auto download + IR convert + cache, task auto-detection, input auto-routing
(image / `.npy` / `.wav`), INT8 quantization.
:::
::::

ovkit is **Apache-2.0** and stays license-clean: it never bundles or downloads
AGPL-licensed model stacks or non-commercial weights. See the
[license policy](usage.md#license-policy).

```{toctree}
:hidden:

usage
guide
cookbook
models
api
```

```{toctree}
:hidden:
:caption: Project

genindex
GitHub <https://github.com/leeyunjai82/ovkit>
```
