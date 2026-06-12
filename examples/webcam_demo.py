#!/usr/bin/env python3
"""Live webcam demo for ovkit — FastAPI + uvicorn.

Pick a model from the dropdown; the laptop webcam is read server-side with
OpenCV, run through ovkit, annotated, and streamed to the browser as MJPEG.

Run::

    pip install fastapi uvicorn
    pip install -e .                      # ovkit, from the repo root
    python examples/webcam_demo.py        # then open http://127.0.0.1:8000

Only runnable vision tasks (detect / classify / segment / pose) are listed.
Register the mirror models first so the dropdown is populated::

    python scripts/build_mirror.py --omz-intel --emit-manifest src/ovkit/manifests/omz.yaml
"""

from __future__ import annotations

import threading

import cv2
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

from ovkit import Model
from ovkit.core.registry import list_models, resolve

RUNNABLE = {"detect", "classify", "segment", "pose"}

app = FastAPI(title="ovkit webcam demo")

_models: dict[str, Model] = {}
_models_lock = threading.Lock()
_cap: cv2.VideoCapture | None = None
_cap_lock = threading.Lock()
# Only one stream is active at a time; bumping the id stops the previous one.
_active = {"id": 0}
_active_lock = threading.Lock()


def runnable_models() -> list[tuple[str, str]]:
    """Return ``(name, task)`` for every registered, runnable vision model."""
    out: list[tuple[str, str]] = []
    for name in list_models():
        try:
            entry = resolve(name)
        except Exception:
            continue
        if entry and entry.task in RUNNABLE:
            out.append((name, entry.task))
    return out


def get_model(name: str) -> Model:
    with _models_lock:
        if name not in _models:
            _models.clear()  # keep only the active model resident (free the old)
            _models[name] = Model(name, device="AUTO")
        return _models[name]


def get_cap() -> cv2.VideoCapture:
    global _cap
    if _cap is None or not _cap.isOpened():
        _cap = cv2.VideoCapture(0)  # laptop webcam
    return _cap


def frames(model_name: str, conf: float, stream_id: int):
    model = get_model(model_name)
    cap = get_cap()
    while _active["id"] == stream_id:  # a newer stream supersedes this one
        with _cap_lock:
            ok, frame = cap.read()
        if not ok:
            break
        try:
            results = model(frame, conf=conf)
            annotated = results[0].plot() if results else frame
        except Exception as exc:  # show the error on the frame, keep streaming
            annotated = frame.copy()
            cv2.putText(
                annotated,
                str(exc)[:80],
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        ok, buf = cv2.imencode(".jpg", annotated)
        if not ok:
            continue
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"


@app.get("/stream")
def stream(model: str, conf: float = 0.25) -> StreamingResponse:
    with _active_lock:
        _active["id"] += 1
        stream_id = _active["id"]
    return StreamingResponse(
        frames(model, conf, stream_id), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    models = runnable_models()
    by_task: dict[str, list[str]] = {}
    for name, task in models:
        by_task.setdefault(task, []).append(name)
    options = ""
    for task in ("detect", "classify", "segment", "pose"):
        names = by_task.get(task)
        if not names:
            continue
        opts = "".join(f'<option value="{n}">{n}</option>' for n in names)
        options += f'<optgroup label="{task} ({len(names)})">{opts}</optgroup>'
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>ovkit webcam demo</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 24px; background: #111; color: #eee; }}
  select, input {{ font-size: 16px; padding: 6px; }}
  #view {{ margin-top: 16px; max-width: 100%; border: 1px solid #333; border-radius: 8px; }}
  label {{ margin-right: 12px; }}
</style></head>
<body>
  <h2>ovkit webcam demo</h2>
  <label>Model:
    <select id="model">{options or '<option>(no models registered)</option>'}</select>
  </label>
  <label>conf: <input id="conf" type="number" value="0.25" min="0" max="1" step="0.05"></label>
  <div><img id="view" src="" alt="stream"></div>
<script>
  const view = document.getElementById('view');
  const model = document.getElementById('model');
  const conf = document.getElementById('conf');
  function refresh() {{
    if (!model.value) return;
    view.src = `/stream?model=${{encodeURIComponent(model.value)}}&conf=${{conf.value}}&t=${{Date.now()}}`;
  }}
  model.addEventListener('change', refresh);
  conf.addEventListener('change', refresh);
  refresh();
</script>
</body></html>"""


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
