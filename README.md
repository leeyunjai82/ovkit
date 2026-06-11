# ovkit

**Use OpenVINO as easily as Ultralytics.** One import, one `Model` class, a
callable object, and clean `Results` — with OpenVINO's strengths layered on top:
`AUTO`/`NPU` devices, async throughput, and INT8 quantization.

```python
from ovkit import Model

model = Model("rtdetr_r50")                 # name -> auto download / convert / cache
results = model("img.jpg", conf=0.25)       # __call__ == predict

for r in results:
    print(r.boxes.xyxy, r.boxes.conf, r.boxes.cls)
    r.save("out.jpg")
```

> **License-clean by design.** ovkit ships and depends only on permissive
> (Apache-2.0/MIT/BSD) models and libraries. AGPL stacks (YOLO/Ultralytics) and
> non-commercial weights (InsightFace pretrained) are intentionally **never**
> bundled or downloaded. See [License policy](#license-policy).

---

## Install

```bash
pip install ovkit                 # core (lightweight)

pip install "ovkit[quant]"        # + NNCF INT8 quantization
pip install "ovkit[genai]"        # + openvino-genai / optimum-intel
pip install "ovkit[anomaly]"      # + anomalib
pip install "ovkit[all]"          # everything
```

Requires Python 3.10+. Core dependencies: `openvino`, `numpy`,
`opencv-python-headless`, `pillow`, `pyyaml`, `huggingface_hub`.

## Quick start

```python
from ovkit import Model

# Load by registered name (downloaded + converted to OpenVINO IR on first use),
# by IR path, or by ONNX path (converted on the fly).
model = Model("rtdetr_r50")
model = Model("path/to/model.xml")
model = Model("path/to/model.onnx")

# Predict on an image path, ndarray, folder, video file, or camera index (int).
results = model("img.jpg", device="NPU", conf=0.25)
results = model.predict("frames/", imgsz=640)
for r in model.predict(0, stream=True):      # webcam, lazy generator
    boxes = r.boxes                          # xyxy, conf, cls
    annotated = r.plot()                     # -> ndarray
```

### Results

| Attribute        | Task      | Contents                              |
| ---------------- | --------- | ------------------------------------- |
| `r.boxes`        | detect    | `xyxy`, `xywh`, `conf`, `cls`         |
| `r.masks`        | segment   | `(N, H, W)` masks                     |
| `r.keypoints`    | pose      | `(N, K, 3)` `[x, y, conf]`            |
| `r.probs`        | classify  | `top1`, `top5`, raw probabilities     |
| `r.plot()`       | all       | annotated `ndarray`                   |
| `r.save(path)`   | all       | render + write to disk                |

### Devices

`device="AUTO"` (default) lets OpenVINO choose; `"CPU"`, `"GPU"`, and `"NPU"`
(Intel® Core™ Ultra and similar) are explicit. Device can be set on the `Model`
or overridden per call. Single images run synchronously; `stream=True` uses an
`AsyncInferQueue` for throughput on video and large folders.

### Quantization (INT8)

```python
model.quantize(calib_images, preset="int8")   # NNCF PTQ; INT8 IR is cached
```

Needs `pip install "ovkit[quant]"`.

## CLI

```bash
ovkit list                 # registered models
ovkit info rtdetr_r50      # source, task, license
ovkit download rtdetr_r50  # fetch + convert to IR
ovkit devices              # available OpenVINO devices
```

## Models & the registry

Models live in a YAML manifest (`src/ovkit/manifests/models.yaml`), separate
from code. **Adding a model is a one-line edit** — no Python changes:

```yaml
rtdetr_r50:
  src: hf
  repo: onnx-community/rtdetr_r50vd
  filename: onnx/model.onnx
  task: detect
  precision: fp16
  license: apache-2.0          # required; must be permissive
```

Resolution order for `Model("name")`:

1. A local `.xml`/`.onnx` path is used directly.
2. A cached IR under `$OVKIT_HOME` (or `~/.cache/ovkit/`) is loaded.
3. Otherwise the manifest source is downloaded, converted to IR, cached, loaded.

Robustness: atomic writes (temp + rename), optional `sha256` integrity checks,
convert-once caching keyed by `(name, precision)`, and an offline mode
(`OVKIT_OFFLINE=1`) that serves only from cache.

### Environment variables

| Variable          | Meaning                                            |
| ----------------- | -------------------------------------------------- |
| `OVKIT_HOME`      | Cache root (default `~/.cache/ovkit`)              |
| `OVKIT_OFFLINE`   | `1` blocks the network; cache-only                 |
| `OVKIT_MANIFESTS` | Extra manifest files/dirs (`os.pathsep`-separated) |

## License policy

ovkit is **Apache-2.0** and stays license-clean:

- **No YOLO/Ultralytics** in dependencies or default models (AGPL-3.0).
- Detection defaults are DETR-family (RT-DETR, RT-DETRv2, D-FINE, RF-DETR) —
  all Apache-2.0.
- Face models will come from Apache-2.0 OMZ weights served from the ovkit HF
  mirror `leeyunjai/ovkit-models` (not the deprecated `omz_downloader`).
- **No InsightFace pretrained weights** (SCRFD/ArcFace) — non-commercial;
  architecture reference only.
- Every manifest entry must declare a permissive `license`; non-permissive
  entries are refused at load time.

## Status (v0)

Implemented end-to-end:

- Core runtime: backend, model, results, registry, download, convert, tasks.
- Image ops (resize/letterbox/crop/zoom/color).
- **Detection** with a DETR-family model: download → convert → infer → Results
  → plot.

Interface stubs (`NotImplementedError` + TODO): classify, segment, pose, face,
genai, and solutions (anomaly/OCR/tracking/reid).

## Documentation

Sphinx docs live in `docs/`:

```bash
pip install -r docs/requirements.txt
sphinx-build -b html docs docs/_build/html
```

## License

Apache-2.0 — see [LICENSE](LICENSE).
