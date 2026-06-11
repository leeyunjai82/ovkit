# Usage

## Install

```bash
pip install ovkit                 # core (lightweight)

pip install "ovkit[quant]"        # + NNCF INT8 quantization
pip install "ovkit[genai]"        # + openvino-genai / optimum-intel
pip install "ovkit[anomaly]"      # + anomalib
pip install "ovkit[all]"          # everything
```

Python 3.10+. Core deps: `openvino`, `numpy`, `opencv-python-headless`,
`pillow`, `pyyaml`, `huggingface_hub`.

## Loading a model

```python
from ovkit import Model

model = Model("rtdetr_r50")          # registered name (downloads + converts on first use)
model = Model("path/to/model.xml")   # OpenVINO IR
model = Model("path/to/model.onnx")  # ONNX -> IR on the fly
```

The task (`detect`/`classify`/`segment`/`pose`) is auto-detected from the
manifest, the IR `rt_info`, or the output signature. Override it with
`Model(..., task="detect")`.

## Predicting

```python
results = model("img.jpg", device="NPU", conf=0.25)
results = model.predict("frames/", imgsz=640)          # a folder of images
for r in model.predict(0, stream=True):                # webcam, lazy generator
    ...
```

`source` may be an image path, a `numpy` array, a directory, a video file, or a
camera index (`int`). `stream=True` returns a generator (use it for video and
large folders); otherwise a list of {class}`~ovkit.Results` is returned.

### Working with `Results`

```python
r = results[0]
r.boxes.xyxy      # (N, 4) pixel boxes
r.boxes.conf      # (N,) scores
r.boxes.cls       # (N,) class ids
r.name_for(2)     # "car"
annotated = r.plot()   # -> ndarray
r.save("out.jpg")
```

| Attribute      | Task     | Contents                       |
| -------------- | -------- | ------------------------------ |
| `r.boxes`      | detect   | `xyxy`, `xywh`, `conf`, `cls`  |
| `r.masks`      | segment  | `(N, H, W)` masks              |
| `r.keypoints`  | pose     | `(N, K, 3)` `[x, y, conf]`     |
| `r.probs`      | classify | `top1`, `top5`, probabilities  |

## Devices

`device="AUTO"` (default) lets OpenVINO choose; `"CPU"`, `"GPU"`, and `"NPU"`
are explicit. Set it on the `Model` or override per call. Single images run
synchronously; `stream=True` uses an `AsyncInferQueue` for throughput.

## Quantization (INT8)

```python
model.quantize(calib_images, preset="int8")   # NNCF PTQ; INT8 IR is cached
```

Requires `pip install "ovkit[quant]"`.

## CLI

```bash
ovkit list                 # registered models
ovkit info rtdetr_r50      # source, task, license
ovkit download rtdetr_r50  # fetch + convert to IR
ovkit devices              # available OpenVINO devices
```

## The model registry

Models live in YAML (`src/ovkit/manifests/models.yaml`), separate from code.
Adding a model is a one-line edit:

```yaml
rtdetr_r50:
  src: hf
  repo: onnx-community/rtdetr_r50vd
  filename: onnx/model.onnx
  task: detect
  precision: fp16
  license: apache-2.0      # required; must be permissive
```

`Model("name")` resolves in this order: local path → cached IR → download +
convert + cache. Robustness: atomic writes, optional `sha256` checks,
convert-once caching keyed by `(name, precision)`, and offline mode.

### Environment variables

| Variable          | Meaning                                            |
| ----------------- | -------------------------------------------------- |
| `OVKIT_HOME`      | Cache root (default `~/.cache/ovkit`)              |
| `OVKIT_OFFLINE`   | `1` blocks the network; cache-only                 |
| `OVKIT_MANIFESTS` | Extra manifest files/dirs (`os.pathsep`-separated) |

## License policy

ovkit is Apache-2.0 and stays license-clean:

- **No YOLO/Ultralytics** (AGPL-3.0) in dependencies or default models.
- Detection defaults are DETR-family (RT-DETR, RT-DETRv2, D-FINE, RF-DETR) —
  all Apache-2.0.
- Face models come from Apache-2.0 OMZ weights on the ovkit HF mirror
  `leeyunjai/ovkit-models` (not the deprecated `omz_downloader`).
- **No InsightFace pretrained weights** — non-commercial; architecture
  reference only.
- Every manifest entry must declare a permissive `license`; non-permissive
  entries are refused at load time.
