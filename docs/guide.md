# Guide (end to end)

A complete walkthrough: install ovkit, run any model, and — if you maintain the
mirror — build, validate, and publish the model set. Reads top to bottom; jump
to [Troubleshooting](#guide-troubleshooting) when something misbehaves.

## 1. What ovkit is

ovkit wraps OpenVINO behind one {class}`~ovkit.Model` object. You give it a model
name or file and call it on an image; it returns a clean {class}`~ovkit.Results`.
Download, OpenVINO IR conversion, caching, device compilation, task detection,
and pre/post-processing are automatic. Models are served from a Hugging Face
mirror (`leeyunjai/ovkit-models`); GenAI (LLM/STT) goes through openvino-genai.

## 2. Installation

ovkit is installed from source into a virtual environment:

```bash
git clone https://github.com/leeyunjai82/ovkit.git
cd ovkit
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

pip install -e .                     # core (lightweight)
pip install -e ".[quant]"            # + NNCF INT8 quantization
pip install -e ".[genai]"            # + openvino-genai / optimum-intel
pip install -e ".[all]"              # everything
```

`-e` is an editable install, so `git pull` updates ovkit with no reinstall.
Python 3.10+.

## 3. Quickstart

```python
from ovkit import Model

model = Model("rtdetr_r50")              # name -> auto download / convert / cache
for r in model("street.jpg", conf=0.25): # __call__ == predict
    print(r.boxes.xyxy, r.boxes.conf, r.boxes.cls)
    r.save("out.jpg")
```

## 4. Choosing a model

Three ways to name a model:

- **Capability alias** — friendly defaults: `Model("face_detection")`,
  `Model("pose")`, `Model("llm")`. See the {doc}`models` catalog for the full
  alias table.
- **Registered name** — e.g. `Model("rtdetr_r50")`; resolved from the YAML
  manifests, downloaded from the mirror, converted, and cached.
- **File path** — `Model("model.xml")` (IR) or `Model("model.onnx")` (converted
  on first use).

Discover models from the CLI:

```bash
ovkit list                 # every registered model: name / task / description
ovkit info face_detection  # source, task, license, precision (follows aliases)
```

## 5. Running inference

Calling the model (`model(x)`) equals `model.predict(x)`. **The input type is
auto-detected:**

```python
model("img.jpg")                       # image file
model(cv2.imread("img.jpg"))           # HWC BGR ndarray
model("frames/")                       # a folder of images
model("clip.mp4")                      # a video file
for r in model.predict(0, stream=True):# webcam (camera index) — lazy generator
    annotated = r.plot()
```

- `conf` — confidence threshold for detection/instance tasks.
- `stream=True` — returns a lazy **generator** (process frames one at a time);
  otherwise you get a `list` of {class}`~ovkit.Results`.
- Non-image inputs (`.npy`, `.wav`, non-image ndarray) auto-route to raw
  inference and return `{name: ndarray}` — see [§7](#guide-low-level).

### Results

```python
r = model("img.jpg")[0]
r.boxes.xyxy      # (N,4) pixel boxes; also .xywh .conf .cls
r.name_for(2)     # "car"  (class id -> name)
annotated = r.plot()   # -> annotated ndarray (boxes/masks/keypoints/text)
r.save("out.jpg")
```

| attribute | task | holds |
| --------- | ---- | ----- |
| `r.boxes` | detect | `xyxy`, `xywh`, `conf`, `cls` |
| `r.masks` | segment | `(N,H,W)` masks (or a class map) |
| `r.keypoints` | pose | `(N,K,3)` `[x,y,conf]` |
| `r.probs` | classify | `top1`, `top5`, scores |
| `r.text` | ocr | decoded string |
| `r.tensors` | generic | raw `{name: ndarray}` |

## 6. Task detection

ovkit picks the **task** (decoder) from, in order: the manifest `task` field →
IR `rt_info` → output-shape heuristics. Override it:

```python
Model("some.xml", task="detect")   # detect | classify | segment | pose | ocr
```

Vision tasks get a typed decoder; everything else falls back to a generic
adapter that returns raw output tensors.

(guide-low-level)=
## 7. Low level — all models (NLP / audio / multi-input)

Models that don't take an image (BERT, translation, multi-input face pipelines)
take tensors you build yourself:

```python
m = Model("bert_small_uncased_whole_word_masking_squad_0002")
print(m.inputs)                                 # [(name, shape, dtype), ...]
out = m.infer({"input_ids": ids, "attention_mask": mask})   # {name: ndarray}
```

Grayscale (1-channel) models are handled automatically: the preprocessor makes a
3-channel tensor and the backend reconciles it to the model's channel count, so
OCR and grayscale classifiers just work.

## 8. Devices

`device="AUTO"` (default) lets OpenVINO pick; `"CPU"`/`"GPU"`/`"NPU"` force one.
Set it on the `Model` or per call. A single image runs synchronously;
`stream=True` uses `AsyncInferQueue` for throughput.

```python
from ovkit.core.backend import available_devices
print(available_devices())             # e.g. ['CPU', 'GPU', 'NPU']
Model("rtdetr_r50")("img.jpg", device="GPU")
```

## 9. Quantization (INT8)

```python
m = Model("rtdetr_r50")
m.quantize(["calib1.jpg", "calib2.jpg", ...], preset="int8")   # NNCF PTQ
r = m("img.jpg")                                               # now INT8
```

Requires `pip install -e ".[quant]"`.

## 10. GenAI (LLM / STT)

```python
from ovkit.genai import pipeline

llm = pipeline("tinyllama_chat")               # or Model alias: "llm"
print(llm.generate("Explain OpenVINO in one sentence.", max_new_tokens=64))

stt = pipeline("whisper_base")                 # speech-to-text ("stt")
print(stt.generate(audio_16k_mono_float32))
```

genai models are served from the mirror (subfolder) with the upstream OpenVINO
repo as a fallback. Requires `pip install -e ".[genai]"`.

(guide-mirror)=
## 11. The model mirror (maintainer)

ovkit downloads models from a Hugging Face repo you control
(`leeyunjai/ovkit-models`). The end user never runs the scripts below — they
just `Model("name")`. This section is for whoever populates/refreshes the mirror.

### Pipeline at a glance

```
build_mirror.py  → download + convert + validate + upload  (writes the mirror)
verify_mirror.py → check every model has a complete, sized IR
selfcheck.py     → download from the mirror and actually run each model
                   (--prune-manifest drops anything that won't load)
--emit-manifest  → write src/ovkit/manifests/omz.yaml (the runtime registry)
```

### Build / refresh

```bash
export HF_TOKEN=...                # write token (or: huggingface-cli login)
                                   # Windows PowerShell: $env:HF_TOKEN = "..."

# Mirror the curated models + the whole Apache-2.0 OMZ set:
python scripts/build_mirror.py --omz-intel

# Mirror only genai (whole openvino-genai dirs -> genai/<name>/):
python scripts/build_mirror.py --models tinyllama_chat whisper_base
```

Each model is downloaded, converted to IR, **compile-validated**, and uploaded
with a model card + LICENSE. A model whose weights are empty/missing or whose
source URL is dead is reported as failed and **not** uploaded.

### Generate the runtime manifest

```bash
python scripts/build_mirror.py --omz-intel \
    --emit-manifest src/ovkit/manifests/omz.yaml
```

This writes one entry per OMZ model pointing at the mirror, **cross-checked**
against the mirror (a too-small/missing `.bin` is excluded). genai is kept in
`genai.yaml`, never here.

### Verify + prune

```bash
python scripts/verify_mirror.py            # per-task counts; flags tiny/missing .bin
python scripts/selfcheck.py --prune-manifest src/ovkit/manifests/omz.yaml \
    --load-only --no-genai
```

`selfcheck` downloads every registered model from the mirror and compiles it.
`--prune-manifest` rewrites `omz.yaml` keeping **only models that load**,
dropping ones that fail to download or compile. Use `--load-only` so blank-frame
dummy inference (which doesn't fit every model) isn't mistaken for a failure.

### Publish

`omz.yaml` lives only on your machine until you commit it:

```bash
git add src/ovkit/manifests/omz.yaml
git commit -m "Update OMZ runtime manifest"
git push
```

Once committed, every user resolves those models by name (and the capability
aliases that point at them light up).

## 12. Adding a model

Models are data, not code — add one line to a manifest:

```yaml
my_model:
  src: hf                          # hf | url | genai
  repo: leeyunjai/ovkit-models
  filename: detect/my_model/model.xml
  task: detect
  description: One-line summary shown by `ovkit list`.
  license: apache-2.0             # required; must be permissive
  fallback: { src: hf, repo: onnx-community/..., filename: onnx/model.onnx }
```

Add a capability alias in `aliases.yaml`: `my_alias: { alias: my_model }`.

(guide-troubleshooting)=
## 13. Troubleshooting

| Symptom | Cause & fix |
| ------- | ----------- |
| `Failed to download ... 403` from huggingface.co | Your network/environment can't reach HF. On Claude Code on the web, pick a network policy that allows huggingface.co. |
| `Empty weights data in bin file` / `core.cpp:135` at load | The mirror's `.bin` is empty/corrupt. Re-mirror (`build_mirror.py --omz-intel`); if the source is dead it's reported failed — then `selfcheck --prune-manifest` drops it. |
| OCR / grayscale model fails at inference | Fixed: the backend reconciles 3↔1 channels automatically. Update with `git pull`. |
| `This model takes N inputs ...` | Multi-input model (e.g. gaze). Use `model.infer({...})` with all inputs, not a single image. |
| `... repo is gated` (401) on a public repo | HF "Gated access" is separate from visibility — disable access requests on the model page, or authenticate. |
| Re-generating `omz.yaml` keeps re-adding a broken model | A stale `omz.yaml` overrode `genai.yaml`/curated entries. Delete `omz.yaml`, mirror genai first, then re-emit. |
| `ovkit list` shows a model as `?` | An alias whose target needs `omz.yaml`; generate + commit it ([§11](#guide-mirror)). |

Run the full self-check any time (where HF is reachable):

```bash
python scripts/selfcheck.py            # env, HF, mirror, download+run, genai
```

## 14. CLI reference

```bash
ovkit list                 # registered models: name / task / description
ovkit info <name>          # source, task, license, precision (follows aliases)
ovkit download <name>      # download + IR convert (warm the cache)
ovkit devices              # available OpenVINO devices
```

## 15. Environment variables

| variable | meaning |
| -------- | ------- |
| `OVKIT_HOME` | cache root (default `~/.cache/ovkit`) |
| `OVKIT_OFFLINE` | `1` = cache only, no network |
| `OVKIT_MANIFESTS` | extra manifest paths (`os.pathsep`-separated) |
| `OVKIT_MIRROR` | override the mirror repo id (scripts) |
| `HF_TOKEN` | Hugging Face token for private repos / uploads |
