#!/usr/bin/env python3
"""ovkit model tester — FastAPI + uvicorn (image upload).

Pick a model, upload an image, press Run: the annotated result and a text
summary (boxes / top-5 / mask classes / keypoints / raw tensor shapes) are
shown. No webcam needed — good for testing specific images.

Run::

    pip install -e . && pip install -r examples/requirements.txt
    python scripts/build_mirror.py --omz-intel --emit-manifest src/ovkit/manifests/omz.yaml
    python examples/web_app.py        # open http://127.0.0.1:8000
"""

from __future__ import annotations

import base64
import threading

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from ovkit import Model
from ovkit.core.registry import list_models, resolve

app = FastAPI(title="ovkit model tester")

_models: dict[str, Model] = {}
_lock = threading.Lock()


def get_model(name: str) -> Model:
    with _lock:
        if name not in _models:
            _models[name] = Model(name, device="AUTO")
        return _models[name]


def summarize(r) -> str:
    lines = [f"task: {r.task}"]
    if r.boxes is not None:
        lines.append(f"{len(r.boxes)} detections")
        for x1, y1, x2, y2, c, cl in r.boxes.data[:25]:
            box = f"[{int(x1)},{int(y1)},{int(x2)},{int(y2)}]"
            lines.append(f"  {r.name_for(int(cl))} {c:.2f} {box}")
    if r.probs is not None:
        lines.append("top-5:")
        for i in r.probs.top5:
            lines.append(f"  {r.name_for(int(i))}: {r.probs.data[int(i)]:.3f}")
    if r.masks is not None:
        classes = np.unique(r.masks.data).tolist()
        lines.append(f"mask {tuple(r.masks.data.shape)}, classes {classes[:25]}")
    if r.keypoints is not None:
        lines.append(f"{len(r.keypoints)} instance(s), {r.keypoints.data.shape[1]} keypoints")
    if r.tensors is not None:
        lines.append("raw outputs:")
        for n, a in r.tensors.items():
            lines.append(f"  {n}: {tuple(np.asarray(a).shape)} {np.asarray(a).dtype}")
    return "\n".join(lines)


@app.post("/run")
async def run(model: str = Form(...), conf: float = Form(0.25), file: UploadFile = File(...)):
    data = np.frombuffer(await file.read(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "could not read image"}, status_code=400)
    try:
        results = get_model(model)(img, conf=conf)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)

    r = results[0] if isinstance(results, list) and results else None
    if r is None:  # raw infer returned a dict (non-image model on an image)
        return JSONResponse({"summary": f"raw outputs: {list(results)}", "image": ""})
    ok, buf = cv2.imencode(".jpg", r.plot())
    b64 = base64.b64encode(buf.tobytes()).decode() if ok else ""
    return JSONResponse({"summary": summarize(r), "image": f"data:image/jpeg;base64,{b64}"})


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    by_task: dict[str, list[str]] = {}
    for name in list_models():
        try:
            entry = resolve(name)
        except Exception:
            continue
        if entry:
            by_task.setdefault(entry.task or "other", []).append(name)
    options = ""
    for task in sorted(by_task):
        opts = "".join(f"<option>{n}</option>" for n in by_task[task])
        options += f'<optgroup label="{task} ({len(by_task[task])})">{opts}</optgroup>'
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>ovkit model tester</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 24px; background: #111; color: #eee; }}
  select, input, button {{ font-size: 15px; padding: 6px; }}
  #out {{ display: flex; gap: 16px; margin-top: 16px; flex-wrap: wrap; }}
  #res {{ max-width: 70vw; border: 1px solid #333; border-radius: 8px; }}
  pre {{ background: #1c1c1c; padding: 12px; border-radius: 8px; max-height: 70vh; overflow: auto; }}
  label {{ margin-right: 12px; }}
</style></head>
<body>
  <h2>ovkit model tester</h2>
  <label>Model: <select id="model">{options or '<option>(no models)</option>'}</select></label>
  <label>conf: <input id="conf" type="number" value="0.25" min="0" max="1" step="0.05"></label>
  <label>Image: <input id="file" type="file" accept="image/*"></label>
  <button id="run">Run</button>
  <div id="out">
    <img id="res" src="" alt="">
    <pre id="summary"></pre>
  </div>
<script>
  const $ = (id) => document.getElementById(id);
  $('run').addEventListener('click', async () => {{
    if (!$('file').files[0]) {{ $('summary').textContent = 'pick an image first'; return; }}
    const fd = new FormData();
    fd.append('model', $('model').value);
    fd.append('conf', $('conf').value);
    fd.append('file', $('file').files[0]);
    $('summary').textContent = 'running ' + $('model').value + ' ...';
    const resp = await fetch('/run', {{ method: 'POST', body: fd }});
    const j = await resp.json();
    if (j.error) {{ $('summary').textContent = 'error: ' + j.error; $('res').src = ''; return; }}
    $('res').src = j.image;
    $('summary').textContent = j.summary;
  }});
</script>
</body></html>"""


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
