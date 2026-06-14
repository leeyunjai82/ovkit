# Usage

This guide explains how ovkit works and how to use each part. ovkit wraps
OpenVINO behind a single {class}`~ovkit.Model` object: you give it a model name
or file and call it on an image, and it returns a clean {class}`~ovkit.Results`
container. Everything in between — downloading, converting to OpenVINO IR,
caching, compiling for a device, detecting the task, pre/post-processing — is
handled for you.

## Install

ovkit is not on PyPI yet, so install it from source. A virtual environment keeps
its dependencies isolated from the rest of your system:

```bash
git clone https://github.com/leeyunjai82/ovkit.git
cd ovkit

python -m venv .venv                     # create a virtualenv
source .venv/bin/activate                # Windows: .venv\Scripts\activate

pip install -e .                         # core (lightweight)

pip install -e ".[quant]"                # + NNCF INT8 quantization
pip install -e ".[genai]"                # + openvino-genai / optimum-intel
pip install -e ".[anomaly]"              # + anomalib
pip install -e ".[all]"                  # everything
```

`-e` is an *editable* install: the package points at your checkout, so `git
pull` updates it without reinstalling. Python 3.10+ is required. The core
dependencies are kept light — `openvino`, `numpy`, `opencv-python-headless`,
`pillow`, `pyyaml`, `huggingface_hub` — and heavier pieces (NNCF, openvino-genai,
anomalib) are optional *extras* you opt into with `[...]`.

## Loading a model

```python
from ovkit import Model

model = Model("rtdetr_r50")          # registered name -> auto download + convert + cache
model = Model("path/to/model.xml")   # an OpenVINO IR file
model = Model("path/to/model.onnx")  # an ONNX file -> converted to IR on first use
```

There are three ways to name a model:

- **A registered name** (e.g. `"rtdetr_r50"`) — looked up in the model registry
  (a YAML manifest). On first use the model is downloaded from its source,
  converted to OpenVINO IR if needed, and cached under `~/.cache/ovkit`. Later
  runs load straight from the cache.
- **An IR path** (`.xml`) — used directly.
- **An ONNX path** (`.onnx`) — OpenVINO reads/converts it on load.

### Task auto-detection

ovkit figures out the model's **task** so it can attach the right decoder. It
tries, in order: the `task` field in the manifest → the IR `rt_info` metadata →
a heuristic on the output tensor shapes. If it can't decide, pass it explicitly:

```python
model = Model("some_model.xml", task="detect")   # detect | classify | segment | pose | ocr
```

Vision tasks (`detect`, `classify`, `segment`, `pose`,
`optical_character_recognition`) get a typed decoder; any other task falls back
to a generic adapter that returns the raw output tensors.

## Predicting

Calling the model (`model(x)`) is the same as `model.predict(x)`. The **input
type is auto-detected**:

```python
results = model("img.jpg", device="NPU", conf=0.25)   # an image file
results = model.predict("frames/", imgsz=640)         # a folder of images
results = model.predict("clip.mp4")                   # a video file
for r in model.predict(0, stream=True):               # webcam (camera index)
    annotated = r.plot()
```

- `source` can be an image path, a `numpy` array (HWC BGR), a folder, a video
  file, or a camera index (`int`).
- `conf` is the confidence threshold for detection/instance tasks.
- `stream=True` returns a lazy **generator** (use it for video or large folders
  so frames are processed one at a time); otherwise you get a `list` of
  {class}`~ovkit.Results`.

Non-image inputs are routed to raw inference automatically: a `.npy` tensor, a
`.wav` file, or a non-image `ndarray` is fed straight to the model and the raw
`{name: ndarray}` outputs are returned (see [low-level](#low-level-any-model)).

### Working with `Results`

A `Results` bundles the original image, the task, and whichever output the task
produced. Use `r.plot()` for an annotated image and `r.save(path)` to write it.

```python
r = results[0]
r.boxes.xyxy      # (N, 4) pixel boxes [x1, y1, x2, y2]
r.boxes.conf      # (N,) confidence scores
r.boxes.cls       # (N,) class ids
r.name_for(2)     # "car"  (class id -> name)
annotated = r.plot()   # -> annotated ndarray (boxes/masks/keypoints/text drawn)
r.save("out.jpg")
```

| Attribute      | Task        | Contents                            |
| -------------- | ----------- | ----------------------------------- |
| `r.boxes`      | detect      | `xyxy`, `xywh`, `conf`, `cls`       |
| `r.masks`      | segment     | `(N, H, W)` masks (or 1 class map)  |
| `r.keypoints`  | pose        | `(N, K, 3)` `[x, y, conf]`          |
| `r.probs`      | classify    | `top1`, `top5`, probabilities       |
| `r.text`       | ocr         | decoded string                      |
| `r.tensors`    | generic     | raw `{name: ndarray}` outputs       |

(low-level-any-model)=
## Low-level — any model

For models that don't take an image (NLP / audio / time-series, multiple
inputs), build the input tensors yourself and call {meth}`~ovkit.Model.infer`:

```python
m = Model("bert_small_uncased_whole_word_masking_squad_0002")
print(m.inputs)                              # [(name, shape, dtype), ...]
out = m.infer({"input_ids": ids, "attention_mask": mask})   # {name: ndarray}
```

`m.inputs` tells you exactly what tensors the model expects.

## Devices

`device="AUTO"` (default) lets OpenVINO pick the best available device; `"CPU"`,
`"GPU"`, and `"NPU"` (Intel® Core™ Ultra and similar) are explicit. Set it on the
`Model` or override it per call. Single images run synchronously; `stream=True`
uses an `AsyncInferQueue` for throughput on video and multi-stream workloads.

```python
from ovkit.core.backend import available_devices
print(available_devices())             # e.g. ['CPU', 'GPU', 'NPU']
```

## Quantization (INT8)

Post-training quantization shrinks a model and speeds it up by converting
weights/activations to INT8, using a handful of representative images to
calibrate:

```python
model.quantize(calib_images, preset="int8")   # NNCF PTQ; the INT8 IR is cached
r = model("img.jpg")                           # now served from the INT8 model
```

Requires `pip install -e ".[quant]"`.

## CLI

```bash
ovkit list                 # registered models (name / task / license)
ovkit info rtdetr_r50      # source, task, license, precision
ovkit download rtdetr_r50  # fetch + convert to IR (warm the cache)
ovkit devices              # available OpenVINO devices
```

## The model registry

Models live in YAML manifests (`src/ovkit/manifests/*.yaml`), separate from the
code, so **adding a model is a one-line edit** — no Python changes:

```yaml
rtdetr_r50:
  src: hf                         # hf | url | genai
  repo: leeyunjai/ovkit-models
  filename: detect/rtdetr_r50/model.xml
  task: detect
  precision: fp16
  license: apache-2.0             # required; must be permissive
  fallback:                       # optional: tried if the primary source fails
    src: hf
    repo: onnx-community/rtdetr_r50vd
    filename: onnx/model.onnx
```

`Model("name")` resolves in this order: a local path → a cached IR under
`$OVKIT_HOME` → download from the manifest source → convert to IR → cache. It is
built to be robust: **atomic writes** (download to a temp file, rename on
success), optional **`sha256` integrity** checks, **convert-once caching** keyed
by `(name, precision)`, an **upstream fallback** when the primary source is down,
and an **offline mode**.

### Environment variables

| Variable          | Meaning                                            |
| ----------------- | -------------------------------------------------- |
| `OVKIT_HOME`      | Cache root (default `~/.cache/ovkit`)              |
| `OVKIT_OFFLINE`   | `1` blocks the network; serve from cache only      |
| `OVKIT_MANIFESTS` | Extra manifest files/dirs (`os.pathsep`-separated) |

## GenAI (LLM / STT / TTS)

Modern OpenVINO generative models are handled by **openvino-genai**, re-exposed
through `ovkit.genai` (separate from the vision pipeline). Install the extra and
call `pipeline(name)`:

```python
from ovkit.genai import pipeline

llm = pipeline("tinyllama_chat")          # downloads + builds the pipeline
print(llm.generate("Explain OpenVINO in one sentence.", max_new_tokens=64))

stt = pipeline("whisper_base")            # speech-to-text
print(stt.generate(audio_16k_mono_float32))
```

Registered genai models live in `src/ovkit/manifests/genai.yaml`. Needs
`pip install -e ".[genai]"`.

## License policy

ovkit is Apache-2.0 and stays license-clean:

- **No AGPL-3.0 model stacks** in dependencies or default models.
- Detection defaults are the DETR family (RT-DETR, RT-DETRv2, D-FINE, RF-DETR) —
  all Apache-2.0.
- Face models come from Apache-2.0 OMZ weights on the ovkit HF mirror
  `leeyunjai/ovkit-models` (not the deprecated `omz_downloader`).
- **No InsightFace pretrained weights** — non-commercial; architecture reference
  only.
- Every manifest entry must declare a permissive `license`; non-permissive
  entries are refused at load time.
