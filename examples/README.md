# ovkit examples

## Live webcam demo (`webcam_demo.py`)

A FastAPI + uvicorn page: pick a model from the dropdown and the laptop webcam
runs through ovkit live (boxes / masks / keypoints / class label drawn on the
frame), streamed to the browser as MJPEG.

```bash
pip install -e .            # ovkit, from the repo root
pip install fastapi uvicorn

# register the mirror models so the dropdown is populated (one-time)
python scripts/build_mirror.py --omz-intel --emit-manifest src/ovkit/manifests/omz.yaml

python examples/webcam_demo.py
# open http://127.0.0.1:8000
```

Notes:

- The server (your laptop) reads the webcam via OpenCV (`cv2.VideoCapture(0)`),
  so run it locally. Grant camera permission if your OS asks.
- Only runnable vision tasks are listed: **detect / classify / segment / pose**.
- The first time you pick a model it is downloaded from the mirror and compiled,
  so the first frame can take a few seconds.
- Switching the dropdown reloads the stream with the new model. The `conf` box
  sets the detection confidence threshold.
- If a model's boxes/masks look off, its OMZ preprocessing (channel order /
  mean / size) may differ — add a `preprocess` block to that model's manifest
  entry to tune it.
