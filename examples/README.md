# ovkit examples

```bash
pip install ovkit                        # or: pip install -e . from the repo root
pip install -r examples/requirements.txt # fastapi / uvicorn (web demos only)
```

## Per-task one-file examples

| example | alias used | shows |
| ------- | ---------- | ----- |
| [`detect.py`](detect.py) | `detect` | boxes drawn + printed |
| [`face_analysis.py`](face_analysis.py) | `age_gender` | `"age 31 · male 98%"` |
| [`segment.py`](segment.py) | `segment` | colored class-map overlay |
| [`pose.py`](pose.py) | `pose` | keypoints/skeleton |
| [`classify.py`](classify.py) | `classify` | top-5 classes |
| [`ocr.py`](ocr.py) | `text_recognition` | decoded text |
| [`super_resolution.py`](super_resolution.py) | `super_resolution` | upscaled image |
| [`background_matting.py`](background_matting.py) | `background_matting_mobilenetv2` | cut the subject out (needs frame + empty-scene photo) |

```bash
python examples/detect.py photo.jpg      # each is ~15 lines — read the source
```

Or skip Python entirely:

```bash
ovkit run detect photo.jpg               # prints results, saves photo_out.jpg
```

## `web_app.py` — model tester (all-in-one) ⭐

Pick a model; the **right input appears automatically**:

- **vision** (detect / classify / segment / pose / ocr / ...) → **webcam** (Load
  for a live stream) **or image upload** (Run). Shows the annotated result and a
  summary (detections / top-5 / mask shape / keypoints / OCR text / raw tensors).
- **STT (Whisper) / noise-suppression** → **audio `.wav`** upload.
- **LLM / TTS** → **text** box.

```bash
python examples/web_app.py      # http://127.0.0.1:8000
```

(genai models — LLM/STT/TTS — also need `pip install "ovkit[genai]"`.)

## `webcam_demo.py` — minimal live webcam

A stripped-down live-only version (vision models).

```bash
python examples/webcam_demo.py  # http://127.0.0.1:8000
```

## `denoise_audio.py` — audio (noise suppression)

The mirror's `noise_suppression_*` models are **stateful streaming** models
(audio frame + recurrent states in/out, frame by frame). This example runs that
loop: wav in → denoised wav out.

```bash
python examples/denoise_audio.py noise_suppression_poconetlike_0001 in.wav out.wav
```

Mono 16 kHz wav. (Best-effort generic state loop — if a model pairs states
differently, share the run output and it can be adjusted.)

## GenAI — LLM / STT / TTS (`llm.py`, `stt.py`, `tts.py`)

Modern OpenVINO models via **openvino-genai** (separate from the vision mirror).

```bash
pip install "ovkit[genai]"
python examples/llm.py "Explain OpenVINO in one sentence."   # LLM
python examples/stt.py audio.wav                              # Whisper STT
python examples/tts.py "Hello" /path/to/tts-ov-model out.wav  # TTS (local model dir)
```

Registered genai models live in `src/ovkit/manifests/genai.yaml` (downloaded on
first use). In code:

```python
from ovkit.genai import pipeline
llm = pipeline("tinyllama_chat"); print(llm.generate("Hi", max_new_tokens=50))
stt = pipeline("whisper_base");  print(stt.generate(audio_16k_mono))
```

## `predict.py` — one-shot CLI

```bash
python examples/predict.py rtdetr_r50 photo.jpg --conf 0.25 --save out.jpg
python examples/predict.py face_detection_retail_0005 face.jpg
python examples/predict.py road_segmentation_adas_0001 road.jpg
```

Notes:

- Only vision tasks draw on the image; other models print/return raw tensors.
- If a model's boxes/masks look off, its OMZ preprocessing (channel order /
  mean / size) may differ — add a `preprocess` block to that model's manifest
  entry to tune it.
