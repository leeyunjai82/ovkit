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
    if r.tensors is not None and len(lines) == 1:
        lines.append("raw outputs:")
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


@app.post("/stop")
def stop_stream() -> JSONResponse:
    _active["id"] += 1  # invalidate any running webcam generator
    return JSONResponse({"ok": True})


@app.get("/selfcheck_stream")
def selfcheck_stream(load_only: int = 1) -> StreamingResponse:
    """Test EVERY registered model (download + compile [+ quick inference]);
    stream one JSON line per model as Server-Sent Events for the live table."""
    import json as _json
    import time as _time

    def gen():
        names = [
            n
            for n in list_models()
            if (e := resolve(n)) is not None and e.name == n and e.src != "genai"
        ]
        yield f"data: {_json.dumps({'type': 'start', 'total': len(names)})}\n\n"
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        counts = {"ok": 0, "warn": 0, "fail": 0}
        for name in names:
            entry = resolve(name)
            yield "data: " + _json.dumps({"type": "running", "name": name}) + "\n\n"
            t0 = _time.perf_counter()
            status, detail, full = "ok", "", ""
            device_note = ""
            try:
                with _lock:
                    _models.clear()  # bound memory across the sweep
                try:
                    model = Model(name, device="AUTO")
                    n_inputs = len(model.inputs)  # forces compile
                except Exception:
                    # AUTO can pick GPU/NPU, whose compilers reject models the
                    # CPU plugin handles (dynamic shapes, exotic ops). Retry on
                    # CPU so "works, but only on CPU" isn't reported as broken.
                    with _lock:
                        _models.clear()
                    model = Model(name, device="CPU")
                    n_inputs = len(model.inputs)
                    device_note = " [CPU only — AUTO failed]"
                if load_only or n_inputs > 1 or model_kind(name) != "image":
                    detail = "loaded" + (" (multi-input)" if n_inputs > 1 else "") + device_note
                else:
                    r = model(img)[0]
                    if r.boxes is not None:
                        detail = f"{len(r.boxes)} boxes"
                    elif r.text:
                        detail = f'"{r.text}"'
                    elif r.probs is not None:
                        detail = f"top1 {r.name_for(r.probs.top1)}"
                    elif r.masks is not None:
                        detail = f"masks {tuple(r.masks.data.shape)}"
                    elif r.keypoints is not None:
                        detail = f"keypoints {tuple(r.keypoints.data.shape)}"
                    else:
                        detail = "ran (raw tensors)"
                    detail += device_note
            except Exception as exc:
                msg = " ".join(str(exc).split())
                full = f"{type(exc).__name__}: {msg}"
                if "Failed to download" in msg or "primary source failed" in msg:
                    status, detail = "fail", "download failed"
                else:
                    status, detail = "warn", f"{type(exc).__name__}: {msg[:70]}"
            counts[status] += 1
            yield "data: " + _json.dumps(
                {
                    "type": "model",
                    "name": name,
                    "task": (entry.task if entry else "") or "",
                    "status": status,
                    "detail": detail,
                    "full": full,
                    "ms": round((_time.perf_counter() - t0) * 1000),
                }
            ) + "\n\n"
        yield f"data: {_json.dumps({'type': 'done', **counts})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


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
            if entry.name != name:
                continue  # capability alias — its target is already listed
            by_task.setdefault(entry.task or "other", []).append(name)
            kinds[name] = model_kind(name)
    descs: dict[str, str] = {}
    for names in by_task.values():
        for n in names:
            e = resolve(n)
            descs[n] = (e.description or "") if e else ""
    options = ""
    for task in sorted(by_task):
        opts = "".join(f"<option>{n}</option>" for n in by_task[task])
        options += f'<optgroup label="{task} ({len(by_task[task])})">{opts}</optgroup>'
    total = sum(len(v) for v in by_task.values())
    import json

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>ovkit</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root {{
    --bg:#0d1117; --card:#161b22; --card2:#1c2330; --line:#2a3240;
    --fg:#e6edf3; --muted:#8b949e; --brand:#22b8cf; --brand-dim:#0a7d8c;
    --ok:#3fb950; --warn:#d29922; --fail:#f85149;
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--fg);
         font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }}
  header {{ display:flex; align-items:center; gap:12px; padding:14px 22px;
            border-bottom:1px solid var(--line); background:var(--card); position:sticky; top:0; z-index:5; }}
  .logo {{ width:14px; height:14px; border-radius:4px;
           background:linear-gradient(135deg,var(--brand),var(--brand-dim)); }}
  header b {{ font-size:17px; letter-spacing:.3px; }}
  header .sub {{ color:var(--muted); font-size:13px; }}
  header a {{ color:var(--brand); text-decoration:none; margin-left:auto; font-size:13px; }}
  .tabs {{ display:flex; gap:6px; padding:14px 22px 0; }}
  .tab {{ padding:8px 16px; border-radius:9px 9px 0 0; cursor:pointer; color:var(--muted);
          border:1px solid transparent; border-bottom:none; font-weight:600; font-size:14px; }}
  .tab.active {{ color:var(--fg); background:var(--card); border-color:var(--line); }}
  main {{ padding:0 22px 40px; }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:0 12px 12px 12px; padding:18px; }}
  .row {{ display:flex; gap:12px; align-items:center; flex-wrap:wrap; }}
  select,input[type=file],textarea,input[type=number] {{
    background:var(--card2); color:var(--fg); border:1px solid var(--line);
    border-radius:8px; padding:8px 10px; font-size:14px; }}
  select {{ min-width:340px; }}
  textarea {{ width:100%; resize:vertical; }}
  button {{ background:var(--brand-dim); color:#fff; border:0; border-radius:8px;
            padding:9px 18px; font-size:14px; font-weight:600; cursor:pointer; }}
  button:hover {{ background:var(--brand); }}
  button.ghost {{ background:transparent; border:1px solid var(--line); color:var(--muted); }}
  button.ghost:hover {{ color:var(--fg); border-color:var(--muted); }}
  .seg {{ display:inline-flex; border:1px solid var(--line); border-radius:9px; overflow:hidden; }}
  .seg label {{ padding:7px 16px; cursor:pointer; color:var(--muted); font-size:14px; }}
  .seg input {{ display:none; }}
  .seg label:has(input:checked) {{ background:var(--brand-dim); color:#fff; }}
  .desc {{ color:var(--muted); font-size:13px; margin:6px 2px 0; min-height:18px; }}
  .badge {{ font-size:12px; padding:2px 10px; border-radius:99px;
            background:var(--card2); border:1px solid var(--line); color:var(--brand); }}
  .grid {{ display:grid; grid-template-columns:minmax(0,1.4fr) minmax(280px,1fr); gap:16px; margin-top:16px; }}
  @media (max-width:900px) {{ .grid {{ grid-template-columns:1fr; }} }}
  .media {{ background:#000; border:1px solid var(--line); border-radius:12px;
            min-height:240px; display:flex; align-items:center; justify-content:center; overflow:hidden; }}
  .media img {{ max-width:100%; max-height:70vh; display:block; }}
  .media .hint {{ color:var(--muted); font-size:14px; padding:30px; text-align:center; }}
  .panel {{ background:var(--card2); border:1px solid var(--line); border-radius:12px;
            padding:14px; max-height:70vh; overflow:auto; }}
  .panel h4 {{ margin:0 0 8px; font-size:13px; color:var(--muted);
               text-transform:uppercase; letter-spacing:.6px; }}
  .panel pre {{ margin:0; white-space:pre-wrap; font:13px/1.6 ui-monospace,Consolas,monospace; }}
  .slider {{ display:flex; align-items:center; gap:8px; color:var(--muted); font-size:14px; }}
  input[type=range] {{ accent-color:var(--brand); width:130px; }}
  /* full-test table */
  .bar {{ height:8px; background:var(--card2); border-radius:99px; overflow:hidden;
          border:1px solid var(--line); flex:1; }}
  .bar > div {{ height:100%; width:0%; background:linear-gradient(90deg,var(--brand-dim),var(--brand));
                transition:width .3s; }}
  table {{ width:100%; border-collapse:collapse; margin-top:14px; font-size:14px; }}
  th,td {{ text-align:left; padding:7px 10px; border-bottom:1px solid var(--line); }}
  th {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.5px; }}
  td.mono {{ font-family:ui-monospace,Consolas,monospace; font-size:13px; }}
  .chip {{ display:inline-block; padding:1px 10px; border-radius:99px; font-size:12px; font-weight:700; }}
  .chip.ok {{ background:rgba(63,185,80,.15); color:var(--ok); }}
  .chip.warn {{ background:rgba(210,153,34,.15); color:var(--warn); }}
  .chip.fail {{ background:rgba(248,81,73,.15); color:var(--fail); }}
  .chip.run {{ background:rgba(34,184,207,.15); color:var(--brand); }}
  .muted {{ color:var(--muted); }}
</style></head>
<body>
<header>
  <div class="logo"></div><b>ovkit</b>
  <span class="sub">model tester · {total} models</span>
  <a href="https://leeyunjai82.github.io/ovkit/" target="_blank">docs ↗</a>
</header>

<div class="tabs">
  <div class="tab active" data-tab="one">모델 테스트</div>
  <div class="tab" data-tab="all">전체 전수 테스트</div>
</div>

<main>
<!-- ============ single model ============ -->
<section id="tab-one" class="card">
  <div class="row">
    <select id="model">{options or '<option>(no models)</option>'}</select>
    <span class="badge" id="kind">image</span>
    <span class="seg" id="srcseg">
      <label><input type="radio" name="src" value="webcam" checked>웹캠</label>
      <label><input type="radio" name="src" value="upload">이미지 파일</label>
    </span>
    <span class="slider">conf <input id="conf" type="range" min="0" max="1" step="0.05" value="0.25">
      <span id="confv">0.25</span></span>
  </div>
  <div class="desc" id="desc"></div>

  <div class="row" style="margin-top:12px">
    <span id="webctl"><button id="load">▶ 웹캠 시작</button>
      <button id="stopbtn" class="ghost">■ 정지</button></span>
    <span id="upctl" style="display:none"><input id="file" type="file" accept="image/*">
      <button id="runimg">실행</button></span>
    <span id="audctl" style="display:none"><input id="afile" type="file" accept="audio/wav,.wav">
      <button id="runaudio">실행</button> <span class="muted">(mono 16 kHz .wav)</span></span>
  </div>
  <div id="txtctl" style="display:none; margin-top:12px">
    <textarea id="text" rows="3">Explain OpenVINO in one sentence.</textarea>
    <div style="margin-top:8px"><button id="runtext">실행</button></div>
  </div>

  <div class="grid">
    <div class="media" id="mediabox">
      <span class="hint" id="hint">모델을 고르고 웹캠을 시작하거나 이미지를 올려보세요.<br>
      처음 쓰는 모델은 다운로드 때문에 수십 초 걸릴 수 있어요 (이후엔 캐시).</span>
      <img id="view" src="" alt="" style="display:none">
    </div>
    <div class="panel"><h4>결과</h4><pre id="summary"></pre>
      <audio id="audio" controls style="display:none; width:100%; margin-top:10px"></audio></div>
  </div>
</section>

<!-- ============ full sweep ============ -->
<section id="tab-all" class="card" style="display:none">
  <div class="row">
    <button id="sweep">전체 모델 전수 테스트 시작</button>
    <label class="muted" style="font-size:14px">
      <input type="checkbox" id="loadonly" checked> 로드까지만 (빠름 — 다운로드+컴파일 확인)</label>
    <div class="bar"><div id="prog"></div></div>
    <span class="muted" id="count">0 / 0</span>
  </div>
  <div class="desc">모델을 <b>하나씩 순서대로</b> 미러에서 받아 검사합니다. 처음엔 다운로드 때문에
  느리지만 전부 캐시되므로 두 번째부터는 빠릅니다. 창을 닫아도 다시 열어 이어서 돌리면 됩니다.</div>
  <div class="row" style="margin-top:10px; gap:8px">
    <span class="chip ok" id="n-ok">✓ 0</span>
    <span class="chip warn" id="n-warn">⚠ 0</span>
    <span class="chip fail" id="n-fail">✗ 0</span>
  </div>
  <table id="tbl"><thead><tr>
    <th>#</th><th>model</th><th>task</th><th>status</th><th>detail</th><th>ms</th>
  </tr></thead><tbody></tbody></table>
</section>
</main>

<script>
  const KINDS = {json.dumps(kinds)};
  const DESCS = {json.dumps(descs)};
  const $ = (id) => document.getElementById(id);

  // tabs
  document.querySelectorAll('.tab').forEach(t => t.onclick = () => {{
    document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
    t.classList.add('active');
    $('tab-one').style.display = t.dataset.tab==='one' ? '' : 'none';
    $('tab-all').style.display = t.dataset.tab==='all' ? '' : 'none';
  }});

  // ---- single model ----
  function curKind() {{ return KINDS[$('model').value] || 'image'; }}
  function show(kind) {{
    $('kind').textContent = kind;
    $('desc').textContent = DESCS[$('model').value] || '';
    $('srcseg').style.display = kind==='image' ? '' : 'none';
    const web = kind==='image' && document.querySelector('input[name=src]:checked').value==='webcam';
    $('webctl').style.display = kind==='image' && web ? '' : 'none';
    $('upctl').style.display  = kind==='image' && !web ? '' : 'none';
    $('audctl').style.display = kind==='audio' ? '' : 'none';
    $('txtctl').style.display = kind==='text'  ? '' : 'none';
    stopView();
  }}
  function stopView() {{
    $('view').src=''; $('view').style.display='none'; $('hint').style.display='';
    $('audio').style.display='none'; $('summary').textContent='';
    fetch('/stop', {{method:'POST'}});
  }}
  function showImg(src) {{ $('hint').style.display='none'; $('view').style.display=''; $('view').src=src; }}
  $('model').onchange = () => show(curKind());
  document.querySelectorAll('input[name=src]').forEach(el => el.onchange = () => show(curKind()));
  $('conf').oninput = () => $('confv').textContent = $('conf').value;

  async function post(url, fd) {{
    $('summary').textContent = $('model').value + ' 실행 중... (처음이면 다운로드에 시간이 걸립니다)';
    try {{
      const j = await (await fetch(url,{{method:'POST',body:fd}})).json();
      if (j.error) {{ $('summary').textContent = '오류: ' + j.error; return; }}
      $('summary').textContent = j.summary || '';
      if (j.image) showImg(j.image);
      if (j.audio) {{ $('audio').src=j.audio; $('audio').style.display=''; }}
    }} catch (e) {{ $('summary').textContent = '요청 실패: ' + e; }}
  }}
  $('load').onclick = () => {{
    $('summary').textContent = '스트리밍 중... (모델 첫 로드시 수십 초)';
    showImg(`/stream?model=${{encodeURIComponent($('model').value)}}&conf=${{$('conf').value}}&t=${{Date.now()}}`);
  }};
  $('stopbtn').onclick = stopView;
  $('runimg').onclick = () => {{
    if(!$('file').files[0]) return; const fd=new FormData();
    fd.append('model',$('model').value); fd.append('conf',$('conf').value);
    fd.append('file',$('file').files[0]); post('/run', fd);
  }};
  $('runaudio').onclick = () => {{
    if(!$('afile').files[0]) return; const fd=new FormData();
    fd.append('model',$('model').value); fd.append('file',$('afile').files[0]);
    post('/run_audio', fd);
  }};
  $('runtext').onclick = () => {{
    const fd=new FormData(); fd.append('model',$('model').value);
    fd.append('text',$('text').value); post('/run_text', fd);
  }};
  show(curKind());

  // ---- full sweep ----
  let es = null, done = 0, total = 0;
  const C = {{ok:0, warn:0, fail:0}};
  $('sweep').onclick = () => {{
    if (es) {{ es.close(); }}
    $('tbl').querySelector('tbody').innerHTML=''; done=0; C.ok=C.warn=C.fail=0; refresh();
    $('sweep').textContent='실행 중...'; $('sweep').disabled=true;
    es = new EventSource('/selfcheck_stream?load_only=' + ($('loadonly').checked?1:0));
    es.onmessage = (ev) => {{
      const d = JSON.parse(ev.data);
      if (d.type==='start') {{ total=d.total; refresh(); }}
      else if (d.type==='running') {{ setRow(d.name, '', 'run', '다운로드/컴파일 중...', ''); }}
      else if (d.type==='model') {{
        done++; C[d.status]++;
        setRow(d.name, d.task, d.status, d.detail, d.ms+' ms', d.full); refresh();
      }}
      else if (d.type==='done') {{
        es.close(); es=null;
        $('sweep').textContent='전체 모델 전수 테스트 시작'; $('sweep').disabled=false;
      }}
    }};
    es.onerror = () => {{ if(es) es.close(); es=null;
      $('sweep').textContent='전체 모델 전수 테스트 시작'; $('sweep').disabled=false; }};
  }};
  function refresh() {{
    $('count').textContent = done + ' / ' + (total||'?');
    $('prog').style.width = total ? (100*done/total)+'%' : '0%';
    $('n-ok').textContent='✓ '+C.ok; $('n-warn').textContent='⚠ '+C.warn; $('n-fail').textContent='✗ '+C.fail;
  }}
  const ICONS = {{ok:'✓ OK', warn:'⚠ 확인필요', fail:'✗ 실패', run:'… 진행중'}};
  function setRow(name, task, status, detail, ms, full) {{
    let tr = document.getElementById('row-'+name);
    if (!tr) {{
      tr = document.createElement('tr'); tr.id='row-'+name;
      tr.innerHTML = '<td class="muted"></td><td class="mono"></td><td class="muted"></td><td></td><td class="muted"></td><td class="muted"></td>';
      $('tbl').querySelector('tbody').appendChild(tr);
      tr.children[0].textContent = $('tbl').querySelectorAll('tbody tr').length;
      tr.children[1].textContent = name;
    }}
    if (task) tr.children[2].textContent = task;
    tr.children[3].innerHTML = '<span class="chip '+status+'">'+ICONS[status]+'</span>';
    tr.children[4].textContent = detail; tr.children[5].textContent = ms;
    if (full) {{                     // full error: hover for all of it, click to expand
      tr.children[4].title = full;
      tr.children[4].style.cursor = 'help';
      tr.children[4].onclick = () => {{ tr.children[4].textContent = full; }};
    }}
    tr.scrollIntoView({{block:'nearest'}});
  }}
</script>
</body></html>"""


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
