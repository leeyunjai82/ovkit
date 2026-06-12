# ovkit examples

```bash
pip install -e .                         # ovkit, from the repo root
pip install -r examples/requirements.txt # fastapi / uvicorn / python-multipart

# register the mirror models so the dropdowns are populated (one-time)
python scripts/build_mirror.py --omz-intel --emit-manifest src/ovkit/manifests/omz.yaml
```

## `web_app.py` — model tester (image upload) ⭐

Pick a model, **upload an image**, press Run. Shows the annotated result and a
text summary (detections / top-5 / mask classes / keypoints / raw tensor
shapes). No webcam needed — best for testing specific images.

```bash
python examples/web_app.py      # http://127.0.0.1:8000
```

## `webcam_demo.py` — live webcam

Pick a model, press Load: the laptop webcam (read server-side via OpenCV) runs
through ovkit live and streams to the browser. Switching models stops the old
one and frees its memory.

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
