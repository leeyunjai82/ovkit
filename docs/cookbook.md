# Cookbook — calling every feature

Copy-paste examples for each ovkit capability. (`pip install -e .`; for genai
add `pip install "ovkit[genai]"`, for quantization `pip install "ovkit[quant]"`.)

## Load a model

```python
from ovkit import Model

m = Model("rtdetr_r50")             # registered name -> auto download/convert/cache
m = Model("path/to/model.xml")      # OpenVINO IR
m = Model("path/to/model.onnx")     # ONNX -> IR on the fly
m = Model("rtdetr_r50", device="NPU")            # AUTO | CPU | GPU | NPU
m = Model("some.xml", task="detect")             # override task auto-detection
m = Model("rtdetr_r50", precision="int8")        # target IR precision
```

## Vision tasks (image in → `Results`)

`model(x)` returns a `list[Results]` (one per image). Use `r.plot()` for an
annotated `ndarray` and `r.save("out.jpg")` to write it.

```python
import cv2

# Detection -> boxes
r = Model("rtdetr_r50")("street.jpg", conf=0.25)[0]
for x1, y1, x2, y2, conf, cls in r.boxes.data:
    print(r.name_for(int(cls)), float(conf), [int(x1), int(y1), int(x2), int(y2)])
print(r.boxes.xyxy, r.boxes.xywh, r.boxes.conf, r.boxes.cls)
r.save("det.jpg")

# Classification -> probs
r = Model("resnet_18")("cat.jpg")[0]
print("top1:", r.name_for(r.probs.top1))
print("top5:", [r.name_for(int(i)) for i in r.probs.top5])

# Semantic segmentation -> masks (1, H, W) class map
r = Model("road_segmentation_adas_0001")("road.jpg")[0]
print(r.masks.data.shape)
cv2.imwrite("seg.jpg", r.plot())          # colorized overlay

# Instance segmentation -> boxes + per-instance masks (N, H, W)
r = Model("instance_segmentation_security_0002")("people.jpg")[0]
print(len(r.boxes), r.masks.data.shape)

# Pose -> keypoints (N, K, 3) = [x, y, conf]
r = Model("human_pose_estimation_0001")("person.jpg")[0]
print(r.keypoints.xy, r.keypoints.conf)

# OCR -> decoded text
r = Model("text_recognition_0012")("word.png")[0]
print(r.text)

# Generic (super-res, embeddings, action, ...) -> raw output tensors
r = Model("single_image_super_resolution_1032")("small.png")[0]
for name, arr in r.tensors.items():
    print(name, arr.shape, arr.dtype)
```

## Inputs (auto-detected)

```python
m = Model("rtdetr_r50")
m("img.jpg")                 # file path
m(cv2.imread("img.jpg"))     # HWC BGR ndarray
m("folder/")                 # every image in a folder
m("clip.mp4")                # a video file
for r in m.predict(0, stream=True):   # webcam (camera index), lazy generator
    annotated = r.plot()

# Non-image inputs are auto-routed to raw inference (returns a dict):
m("features.npy")            # a saved tensor
m("speech.wav")              # a .wav (fitted to the model input length)
```

## Low-level (any model, your own tensors)

For NLP / audio / time-series models that take non-image input, build the
tensors yourself:

```python
m = Model("bert_small_uncased_whole_word_masking_squad_0002")
print(m.inputs)                          # [(name, shape, dtype), ...]
out = m.infer({"input_ids": ids, "attention_mask": mask})   # {name: ndarray}
```

## Quantization (INT8, NNCF)

```python
m = Model("rtdetr_r50")
m.quantize(["calib1.jpg", "calib2.jpg", ...], preset="int8")   # caches INT8 IR
r = m("img.jpg")             # now served from the INT8 model
```

## Devices

```python
from ovkit.core.backend import available_devices
print(available_devices())                # ['CPU', 'GPU', 'NPU', ...]
Model("rtdetr_r50")("img.jpg", device="GPU")
```

## GenAI (LLM / STT / TTS) — `ovkit[genai]`

```python
from ovkit.genai import pipeline

llm = pipeline("tinyllama_chat")                       # downloads + builds
print(llm.generate("Explain OpenVINO in one sentence.", max_new_tokens=64))

stt = pipeline("whisper_base")                         # speech-to-text
print(stt.generate(audio_16k_mono_float32))

# A local OpenVINO-genai model directory (pass the pipeline type):
tts = pipeline("/path/to/tts-ov", pipeline_type="text2speech")
```

## CLI

```bash
ovkit list                 # registered models (name / task / license)
ovkit info rtdetr_r50      # source, task, license, precision
ovkit download rtdetr_r50  # fetch + convert to IR
ovkit devices              # OpenVINO devices
```

## Registry / manifest

```python
from ovkit.core.registry import list_models, resolve
print(list_models())
e = resolve("rtdetr_r50")
print(e.task, e.license, e.repo)
```

Add a model with one YAML line in `src/ovkit/manifests/*.yaml`:

```yaml
my_model:
  src: hf                       # hf | url | genai
  repo: leeyunjai/ovkit-models
  filename: detect/my_model/model.xml
  task: detect
  license: apache-2.0
  fallback: { src: hf, repo: onnx-community/..., filename: onnx/model.onnx }
```

Environment: `OVKIT_HOME` (cache), `OVKIT_OFFLINE=1` (cache-only),
`OVKIT_MANIFESTS` (extra manifest paths).

## Mirror tooling (scripts/)

```bash
# Mirror models to your HF repo (OMZ + curated), then verify:
python scripts/build_mirror.py --repo leeyunjai/ovkit-models --omz-intel
python scripts/verify_mirror.py
# Generate a runtime manifest pointing at the mirror:
python scripts/build_mirror.py --omz-intel --emit-manifest src/ovkit/manifests/omz.yaml
```

## Example apps (examples/)

```bash
pip install -r examples/requirements.txt
python examples/web_app.py                 # image upload / webcam / audio / text
python examples/predict.py rtdetr_r50 img.jpg --save out.jpg
python examples/denoise_audio.py noise_suppression_poconetlike_0001 in.wav out.wav
python examples/llm.py "Hello"             # needs ovkit[genai]
python examples/stt.py speech.wav
```
