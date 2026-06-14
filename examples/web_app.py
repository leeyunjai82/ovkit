#!/usr/bin/env python3
"""ovkit model tester — FastAPI + uvicorn (image / webcam / audio / text).

Pick a model; the right input appears automatically:
  * vision (detect/classify/segment/pose/ocr/...) -> **image upload or webcam**
  * STT (Whisper) / noise-suppression                -> **audio (.wav)**
  * LLM / TTS / NLP                                   -> **text**

Run::

    pip install -e . && pip install -r examples/requirements.txt
    # genai (LLM/STT/TTS) also needs:  pip install -e ".[genai]"
    python scripts/build_mirror.py --omz-intel --emit-manifest src/ovkit/manifests/omz.yaml
    python examples/web_app.py        # http://127.0.0.1:8000
"""

from __future__ import annotations

import base64
import io
import threading
import wave

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from ovkit import Model
from ovkit.core.registry import list_models, resolve

app = FastAPI(title="ovkit model tester")

_VISION = {"detect", "classify", "segment", "pose", "optical_character_recognition", "ocr"}
_GENAI_KIND = {"llm": "text", "whisper": "audio", "text2speech": "text", "vlm": "image"}

_models: dict[str, Model] = {}
_genai: dict[str, object] = {}
_lock = threading.Lock()
_cap: cv2.VideoCapture | None = None
_cap_lock = threading.Lock()
_active = {"id": 0}


def model_kind(name: str) -> str:
    try:
        e = resolve(name)
    except Exception:
        return "image"
    if e is None:
        return "image"
    if e.src == "genai":
        return _GENAI_KIND.get(e.extra.get("pipeline"), "text")
    t = e.task or ""
    if t in _VISION or t.startswith("face") or t in {"image_processing", "action_recognition"}:
        return "image"
    if "noise" in t or "audio" in t or "speech" in t:
        return "audio"
    return "text"


def get_model(name: str) -> Model:
    with _lock:
        if name not in _models:
            _models.clear()
            _models[name] = Model(name, device="AUTO")
        return _models[name]


def get_cap() -> cv2.VideoCapture | None:
    """Open the webcam, trying DirectShow first (Windows MSMF often fails)."""
    global _cap
    if _cap is not None and _cap.isOpened():
        return _cap
    candidates = [(0, cv2.CAP_DSHOW), (0, None), (1, cv2.CAP_DSHOW), (1, None)]
    for idx, backend in candidates:
        cap = cv2.VideoCapture(idx) if backend is None else cv2.VideoCapture(idx, backend)
        if cap.isOpened() and cap.read()[0]:
            _cap = cap
            return _cap
        cap.release()
    return None


def _error_frame(msg: str) -> bytes:
    """Render an error message as a JPEG frame so failures are visible."""
    words, lines, cur = msg.split(), [], ""
    for word in words:
        if len(cur) + len(word) > 58:
            lines.append(cur)
            cur = word
        else:
            cur = (cur + " " + word).strip()
    lines.append(cur)
    height = max(360, 30 + len(lines) * 26)
    img = np.zeros((height, 720, 3), dtype=np.uint8)
    for i, line in enumerate(lines[:24]):
        cv2.putText(img, line, (10, 30 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 255), 1)
    ok, buf = cv2.imencode(".jpg", img)
    return buf.tobytes() if ok else b""


def read_wav(data: bytes) -> tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(data), "rb") as wf:
        sr = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0, sr


def wav_b64(audio: np.ndarray, sr: int) -> str:
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()


def summarize(r) -> str:
    lines = [f"task: {r.task}"]
    if getattr(r, "text", None):
        lines.append(f'text: "{r.text}"')
    if r.boxes is not None:
        lines.append(f"{len(r.boxes)} detections")
        for x1, y1, x2, y2, c, cl in r.boxes.data[:25]:
            lines.append(
                f"  {r.name_for(int(cl))} {c:.2f} [{int(x1)},{int(y1)},{int(x2)},{int(y2)}]"
            )
    if r.probs is not None:
        lines.append(
            "top-5: "
            + ", ".join(f"{r.name_for(int(i))} {r.probs.data[int(i)]:.2f}" for i in r.probs.top5)
        )
    if r.masks is not None:
        lines.append(f"masks {tuple(r.masks.data.shape)}")
    if r.keypoints is not None:
        lines.append(f"{len(r.keypoints)} instance(s), {r.keypoints.data.shape[1]} keypoints")
    if r.tensors is not None:
        for n, a in r.tensors.items():
            lines.append(f"  {n}: {tuple(np.asarray(a).shape)}")
    return "\n".join(lines)


# --- webcam (live MJPEG) ---------------------------------------------------


def frames(name: str, conf: float, sid: int):
    boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    try:
        model = get_model(name)
    except Exception as exc:  # make the failure visible instead of a dead stream
        yield boundary + _error_frame(f"model load failed: {exc}") + b"\r\n"
        return
    cap = get_cap()
    if cap is None:
        yield (
            boundary
            + _error_frame(
                "webcam not available - close other apps using the camera, and check "
                "Windows camera privacy settings (allow desktop apps)."
            )
            + b"\r\n"
        )
        return
    while _active["id"] == sid:
        with _cap_lock:
            ok, frame = cap.read()
        if not ok:
            break
        try:
            res = model(frame, conf=conf)
            annotated = res[0].plot() if isinstance(res, list) and res else frame
        except Exception as exc:
            annotated = frame.copy()
            cv2.putText(
                annotated, str(exc)[:80], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
            )
        ok, buf = cv2.imencode(".jpg", annotated)
        if ok:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"


@app.get("/stream")
def stream(model: str, conf: float = 0.25) -> StreamingResponse:
    _active["id"] += 1
    return StreamingResponse(
        frames(model, conf, _active["id"]), media_type="multipart/x-mixed-replace; boundary=frame"
    )


# --- image / audio / text handlers -----------------------------------------


@app.post("/run")
async def run_image(model: str = Form(...), conf: float = Form(0.25), file: UploadFile = File(...)):
    data = np.frombuffer(await file.read(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "could not read image"}, status_code=400)
    try:
        res = get_model(model)(img, conf=conf)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)
    r = res[0] if isinstance(res, list) and res else None
    if r is None:
        return JSONResponse({"summary": f"raw outputs: {list(res)}", "image": ""})
    ok, buf = cv2.imencode(".jpg", r.plot())
    b64 = base64.b64encode(buf.tobytes()).decode() if ok else ""
    return JSONResponse({"summary": summarize(r), "image": f"data:image/jpeg;base64,{b64}"})


@app.post("/run_audio")
async def run_audio(model: str = Form(...), file: UploadFile = File(...)):
    try:
        audio, sr = read_wav(await file.read())
    except Exception as exc:
        return JSONResponse({"error": f"wav read failed: {exc}"}, status_code=400)
    kind = model_kind(model)
    try:
        if kind == "audio" and resolve(model) and resolve(model).src == "genai":
            from ovkit.genai import pipeline

            stt = _genai.get(model) or _genai.setdefault(model, pipeline(model))
            return JSONResponse({"summary": f'text: "{stt.generate(audio)}"'})
        # OMZ noise-suppression: streaming state loop.
        out = _denoise(model, audio)
        return JSONResponse({"summary": f"denoised {len(out)} samples", "audio": wav_b64(out, sr)})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.post("/run_text")
async def run_text(model: str = Form(...), text: str = Form(...)):
    try:
        from ovkit.genai import pipeline

        entry = resolve(model)
        if entry is None:
            return JSONResponse(
                {
                    "summary": f"'{model}' is not registered (manifest missing? run git pull "
                    f"and restart the server)."
                }
            )
        if entry.src != "genai":
            return JSONResponse(
                {
                    "summary": f"'{model}' (src={entry.src}, task={entry.task}) needs tokenized "
                    f"input — use model.infer() with your own tensors."
                }
            )
        ptype = entry.extra.get("pipeline")
        print(f"[run_text] {model}: loading genai pipeline ({ptype})...", flush=True)
        pipe = _genai.get(model) or _genai.setdefault(model, pipeline(model))
        if ptype == "text2speech":
            res = pipe.generate(text)
            audio = np.asarray(getattr(res, "speeches", [res])[0]).reshape(-1)
            return JSONResponse({"summary": "synthesized speech", "audio": wav_b64(audio, 16000)})
        out = str(pipe.generate(text, max_new_tokens=200))
        print(f"[run_text] {model} -> {out[:200]}", flush=True)
        return JSONResponse({"summary": out})
    except Exception as exc:
        import traceback

        traceback.print_exc()
        return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=500)


def _denoise(model: str, audio: np.ndarray) -> np.ndarray:
    m = get_model(model)
    info = {n: tuple(s) for n, s, _ in m.inputs}
    audio_name = "input" if "input" in info else max(info, key=lambda n: int(np.prod(info[n])))
    patch = int(info[audio_name][-1])
    state_in = sorted(n for n in info if n != audio_name)
    states = {n: np.zeros(info[n], dtype=np.float32) for n in state_in}
    out, audio_out, state_out = [], None, []
    for i in range(0, len(audio), patch):
        chunk = np.pad(audio[i : i + patch], (0, max(0, patch - len(audio[i : i + patch]))))
        res = m.infer({audio_name: chunk[None].astype(np.float32), **states})
        if audio_out is None:
            audio_out = "output" if "output" in res else next(iter(res))
            state_out = sorted(k for k in res if k != audio_out)
        out.append(np.asarray(res[audio_out]).reshape(-1)[:patch])
        states = {n: np.asarray(res[o]) for n, o in zip(state_in, state_out, strict=False)}
    return np.concatenate(out)[: len(audio)] if out else audio


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    by_task: dict[str, list[str]] = {}
    kinds: dict[str, str] = {}
    for name in list_models():
        try:
            entry = resolve(name)
        except Exception:
            continue
        if entry:
            by_task.setdefault(entry.task or "other", []).append(name)
            kinds[name] = model_kind(name)
    options = ""
    for task in sorted(by_task):
        opts = "".join(f"<option>{n}</option>" for n in by_task[task])
        options += f'<optgroup label="{task} ({len(by_task[task])})">{opts}</optgroup>'
    import json

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>ovkit model tester</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 24px; background:#111; color:#eee; }}
  select, input, button, textarea {{ font-size: 15px; padding: 6px; }}
  #out {{ display:flex; gap:16px; margin-top:16px; flex-wrap:wrap; }}
  #view {{ max-width:70vw; border:1px solid #333; border-radius:8px; }}
  pre {{ background:#1c1c1c; padding:12px; border-radius:8px; max-height:70vh; overflow:auto; }}
  label {{ margin-right:12px; }} .ctl {{ margin-top:10px; }}
</style></head>
<body>
  <h2>ovkit model tester</h2>
  <label>Model: <select id="model">{options or '<option>(no models)</option>'}</select></label>
  <span id="kind" style="color:#9b9"></span>

  <div class="ctl" id="c-image">
    <label><input type="radio" name="src" value="webcam" checked> Webcam</label>
    <label><input type="radio" name="src" value="upload"> Image file</label>
    <label>conf <input id="conf" type="number" value="0.25" min="0" max="1" step="0.05" style="width:70px"></label>
    <span id="webctl"><button id="load">Load</button></span>
    <span id="upctl" style="display:none"><input id="file" type="file" accept="image/*"><button id="runimg">Run</button></span>
  </div>
  <div class="ctl" id="c-audio" style="display:none">
    <input id="afile" type="file" accept="audio/wav,.wav"><button id="runaudio">Run</button>
    <span style="color:#999">(mono 16 kHz .wav)</span>
  </div>
  <div class="ctl" id="c-text" style="display:none">
    <textarea id="text" rows="3" cols="60">Explain OpenVINO in one sentence.</textarea>
    <button id="runtext">Run</button>
  </div>

  <div id="out"><img id="view" src="" alt=""><audio id="audio" controls style="display:none"></audio><pre id="summary"></pre></div>
<script>
  const KINDS = {json.dumps(kinds)};
  const $ = (id) => document.getElementById(id);
  function show(kind) {{
    $('c-image').style.display = kind==='image' ? '' : 'none';
    $('c-audio').style.display = kind==='audio' ? '' : 'none';
    $('c-text').style.display  = kind==='text'  ? '' : 'none';
    $('kind').textContent = ' [' + kind + ']';
    $('view').src=''; $('audio').style.display='none'; $('summary').textContent='';
  }}
  function curKind() {{ return KINDS[$('model').value] || 'image'; }}
  $('model').addEventListener('change', () => show(curKind()));
  document.querySelectorAll('input[name=src]').forEach(el => el.addEventListener('change', () => {{
    const web = document.querySelector('input[name=src]:checked').value==='webcam';
    $('webctl').style.display = web?'':'none'; $('upctl').style.display = web?'none':'';
    $('view').src='';
  }}));
  async function post(url, fd) {{
    $('summary').textContent='running '+$('model').value+' ...';
    const j = await (await fetch(url,{{method:'POST',body:fd}})).json();
    if (j.error) {{ $('summary').textContent='error: '+j.error; return; }}
    $('summary').textContent = j.summary || '';
    if (j.image) {{ $('view').src=j.image; }}
    if (j.audio) {{ $('audio').src=j.audio; $('audio').style.display=''; }}
  }}
  $('load').addEventListener('click', () => {{
    $('summary').textContent='streaming ...';
    $('view').src=`/stream?model=${{encodeURIComponent($('model').value)}}&conf=${{$('conf').value}}&t=${{Date.now()}}`;
  }});
  $('runimg').addEventListener('click', () => {{
    if(!$('file').files[0]) return; const fd=new FormData();
    fd.append('model',$('model').value); fd.append('conf',$('conf').value); fd.append('file',$('file').files[0]);
    post('/run', fd);
  }});
  $('runaudio').addEventListener('click', () => {{
    if(!$('afile').files[0]) return; const fd=new FormData();
    fd.append('model',$('model').value); fd.append('file',$('afile').files[0]); post('/run_audio', fd);
  }});
  $('runtext').addEventListener('click', () => {{
    const fd=new FormData(); fd.append('model',$('model').value); fd.append('text',$('text').value); post('/run_text', fd);
  }});
  show(curKind());
</script>
</body></html>"""


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
