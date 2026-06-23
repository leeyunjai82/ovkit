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
