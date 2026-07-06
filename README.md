# ovkit

**A simple Python inference API for OpenVINO.** One import, one `Model` class, a
callable object, and clean `Results` — with OpenVINO's strengths layered on top:
`AUTO`/`NPU` devices, async throughput, and INT8 quantization.

```python
from ovkit import Model

model = Model("detect")                     # capability alias -> rtdetr_r50
results = model("img.jpg", conf=0.25)       # __call__ == predict

for r in results:
    print(r.boxes.xyxy, r.boxes.conf, r.boxes.cls)
    r.save("out.jpg")
```

**Docs:** https://leeyunjai82.github.io/ovkit/ (한국어: [/ko/](https://leeyunjai82.github.io/ovkit/ko/)) —
usage guide, cookbook, model catalog, API reference.

> **License-clean by design.** ovkit ships and depends only on permissive
> (Apache-2.0/MIT/BSD) models and libraries. AGPL-licensed model stacks and
> non-commercial weights (InsightFace pretrained) are intentionally **never**
> bundled or downloaded. See [License policy](#license-policy).

---

## Install

> Not on PyPI yet — install from source in a virtual environment.

```bash
git clone https://github.com/leeyunjai82/ovkit.git
cd ovkit

python -m venv .venv                     # create a virtualenv
source .venv/bin/activate                # Windows: .venv\Scripts\activate

pip install -e .                         # core (lightweight)

pip install -e ".[quant]"                # + NNCF INT8 quantization
pip install -e ".[genai]"                # + openvino-genai (LLM / STT)
pip install -e ".[all]"                  # everything
```

Requires Python 3.10+. Core dependencies: `openvino`, `numpy`,
`opencv-python-headless`, `pillow`, `pyyaml`, `huggingface_hub`.

## Models: one representative per capability

The registry exposes **35 models — one well-tested model per function** — plus
**capability aliases** so you never pick among variants:

```python
Model("face_detection")("photo.jpg")   # -> face_detection_0205, boxes drawn
Model("age_gender")("face.jpg")        # -> "age 31 · male 98%"
Model("pose")("person.jpg")            # -> keypoints
Model("super_resolution")("small.png") # -> upscaled image from r.plot()

from ovkit.genai import pipeline
pipeline("llm").generate("Hello")      # -> tinyllama_chat (ovkit[genai])
pipeline("stt").generate(audio_16k)    # -> whisper_base
```

Aliases include `detect`, `face_detection`, `person_detection`,
`vehicle_detection`, `text_detection`, `segment`, `instance_segmentation`,
`pose`, `face_landmarks`, `head_pose`, `gaze`, `age_gender`, `emotion`,
`face_reid`, `person_attributes`, `vehicle_attributes`, `classify`,
`image_retrieval`, `super_resolution`, `text_recognition`, `license_plate`,
`qa`, `translation`, `noise_suppression`, `time_series`, `llm`, `stt`.

Every model is served from the ovkit Hugging Face mirror
([`leeyunjai/ovkit-models`](https://huggingface.co/leeyunjai/ovkit-models)),
downloaded on first use and cached. The mirror also hosts the **full**
Apache-2.0 Open Model Zoo set (other accuracy/speed tiers, `int8`, `sparse`
variants) — surfacing a variant is a one-line edit
(see the [model catalog](https://leeyunjai82.github.io/ovkit/models.html)).

`ovkit list` prints the alias table and the model list with descriptions.

## Quick start

```python
from ovkit import Model

# Load by alias, registered name, IR path, or ONNX path.
model = Model("detect")
model = Model("rtdetr_r50")
model = Model("path/to/model.xml")
model = Model("path/to/model.onnx")          # converted to IR on the fly

# Predict on an image path, ndarray, folder, video file, or camera index (int).
results = model("img.jpg", device="NPU", conf=0.25)
results = model.predict("frames/", imgsz=640)
for r in model.predict(0, stream=True):      # webcam, lazy generator
    annotated = r.plot()
```

### Inputs (auto-detected)

`model(x)` routes on the input type — no config needed:

- **image** (path / `ndarray` / folder / video / camera `int`) → vision
  pipeline → typed `Results`.
- **non-image** (`.npy` tensor, `.wav` audio, raw non-image `ndarray`) → fed to
  the model directly → raw `{name: ndarray}`.

Grayscale (1-channel) models and all-image multi-input models (e.g.
super-resolution) are handled automatically. For everything else (NLP / audio
with your own tensors), use the low-level API:

```python
print(model.inputs)            # [(name, shape, dtype), ...]
out = model.infer(tensors)     # {output_name: ndarray}, no preprocessing
```

### Results

| Attribute      | Task            | Contents                                     |
| -------------- | --------------- | -------------------------------------------- |
| `r.boxes`      | detect          | `xyxy`, `xywh`, `conf`, `cls`                |
| `r.masks`      | segment         | `(N, H, W)` masks (or a class map)           |
| `r.keypoints`  | pose, landmarks | `(N, K, 3)` `[x, y, conf]`                   |
| `r.probs`      | classify        | `top1`, `top5`, raw probabilities            |
| `r.text`       | ocr, face attrs | decoded string (`"age 31 · male 98%"`, ...)  |
| `r.tensors`    | generic         | raw `{name: ndarray}` outputs                |
| `r.plot()`     | all             | annotated `ndarray` (or the output image)    |
| `r.save(path)` | all             | render + write to disk                       |

### Devices

`device="AUTO"` (default) lets OpenVINO choose; `"CPU"`, `"GPU"`, and `"NPU"`
(Intel® Core™ Ultra and similar) are explicit — set on the `Model` or per call.
Single images run synchronously; `stream=True` uses an `AsyncInferQueue` for
throughput on video and folders.

### Quantization (INT8)

```python
model.quantize(calib_images, preset="int8")   # NNCF PTQ; INT8 IR is cached
```

Needs `pip install -e ".[quant]"`.

## What runs end-to-end

| Task | Decode | Example models |
| ---- | ------ | -------------- |
| **detect** | DETR / SSD / boxes+labels / YOLO → `boxes` | `rtdetr_r50`, `face_detection_0205`, `person_detection_0202`, ... |
| **classify** | softmax → `probs`; multi-head (type+color) → text | `resnet50_binary_0001`, `vehicle_attributes_...` |
| **segment** | semantic class map + instance masks | `road_segmentation_adas_0001`, `instance_segmentation_person_0007` |
| **pose / landmarks** | heatmap peaks / coord regressors → `keypoints` | `human_pose_estimation_0007`, `landmarks_regression_retail_0009` |
| **face analysis** | age+gender / emotion / head pose / re-id → text & probs | `age_gender_...`, `emotions_...`, `head_pose_...` |
| **OCR** | CTC decode → `r.text` | `text_recognition_0014`, handwritten models |
| **image processing** | output image rendered by `plot()` | `single_image_super_resolution_1033` |
| **GenAI** | openvino-genai pipelines | `tinyllama_chat` (LLM), `whisper_base` (STT) |
| **other** | raw tensors + `model.infer()` | QA (BERT), translation, noise suppression, time series |

## Demo web app

Try any model from the browser — image upload, webcam, audio, and text inputs
appear automatically per model:

```bash
pip install -r examples/requirements.txt
python examples/web_app.py        # http://127.0.0.1:8000
```

More examples in `examples/`: `predict.py`, `webcam_demo.py`,
`denoise_audio.py`, `llm.py`, `stt.py`, `tts.py`.

## CLI

```bash
ovkit list                 # aliases (capability -> model) + models with descriptions
ovkit info face_detection  # source, task, license (follows aliases)
ovkit download detect      # fetch + convert to IR (warm the cache)
ovkit devices              # available OpenVINO devices
```

## The registry

Models live in YAML manifests (`src/ovkit/manifests/`), separate from code.
**Adding a model is a one-line edit** — no Python changes:

```yaml
my_model:
  src: hf                              # hf | url | genai
  repo: leeyunjai/ovkit-models
  filename: detect/my_model/model.xml
  task: detect
  description: One-line summary shown by `ovkit list`.
  license: apache-2.0                  # required; must be permissive
  fallback: { src: hf, repo: onnx-community/..., filename: onnx/model.onnx }
```

Resolution order for `Model("name")`: alias → local path → cached IR under
`$OVKIT_HOME` → download from the primary source → convert → cache → (on
failure) the `fallback` source. Atomic writes, `sha256` checks, convert-once
caching, and an offline mode (`OVKIT_OFFLINE=1`).

### Environment variables

| Variable          | Meaning                                            |
| ----------------- | -------------------------------------------------- |
| `OVKIT_HOME`      | Cache root (default `~/.cache/ovkit`)              |
| `OVKIT_OFFLINE`   | `1` blocks the network; cache-only                 |
| `OVKIT_MANIFESTS` | Extra manifest files/dirs (`os.pathsep`-separated) |
| `OVKIT_MIRROR`    | Override the mirror repo id (scripts)              |

## Maintainer: the model mirror

`scripts/` contains the tooling that populates and verifies the mirror — end
users never need it:

```bash
python scripts/build_mirror.py --omz-intel        # download + convert + validate + upload
python scripts/build_mirror.py --omz-intel --representatives \
    --emit-manifest src/ovkit/manifests/omz.yaml  # regenerate the trimmed registry
python scripts/verify_mirror.py                   # completeness check
python scripts/selfcheck.py                       # download + run every registered model
```

Broken IR (empty weights, dead source URLs) is rejected before upload, and
`selfcheck.py --prune-manifest` drops anything that won't compile. See the
[guide](https://leeyunjai82.github.io/ovkit/guide.html) for the full pipeline.

## License policy

ovkit is **Apache-2.0** and stays license-clean:

- **No AGPL-3.0 model stacks** in dependencies or default models.
- Detection default is DETR-family (`rtdetr_r50`) — Apache-2.0.
- Face models are Apache-2.0 OMZ weights served from the ovkit HF mirror
  `leeyunjai/ovkit-models` (not the deprecated `omz_downloader`).
- **No InsightFace pretrained weights** (SCRFD/ArcFace) — non-commercial;
  architecture reference only.
- Every manifest entry must declare a permissive `license`; non-permissive
  entries are refused at load time.

## Documentation

Published at **https://leeyunjai82.github.io/ovkit/** (English) and
[/ko/](https://leeyunjai82.github.io/ovkit/ko/) (한국어). Build locally:

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html          # EN
sphinx-build -b html docs/ko docs/_build/html/ko    # KO
```

## License

Apache-2.0 — see [LICENSE](LICENSE).
