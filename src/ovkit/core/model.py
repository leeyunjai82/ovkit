"""The single public entry point: :class:`Model`.

``Model`` ties the whole pipeline together — resolve a name/path, download +
convert to IR, compile per device, auto-detect the task, attach the right
adapter, and run prediction — behind a simple callable object::

    from ovkit import Model
    model = Model("rtdetr_r50")
    results = model("img.jpg", conf=0.25)
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..image import ops
from ..recognize import get_adapter
from .backend import Backend
from .constants import class_names
from .convert import to_ir
from .download import fetch
from .errors import ModelNotFoundError, OVKitError
from .i18n import canonical, lang
from .registry import ModelEntry, list_models, resolve
from .results import Results
from .tasks import detect_task

_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
_AUDIO_EXT = {".wav", ".flac", ".ogg", ".mp3", ".m4a"}
#: Sample rate the OMZ audio models are trained at.
DEFAULT_AUDIO_SR = 16_000


class Model:
    """An OpenVINO model with automatic resolution, task detection, and IO.

    One name, one call — and the input can come along in the same call::

        r = Model("detect", "street.jpg")     # runs now -> one Results
        r = Model("얼굴분석", "group.jpg")     # Korean names work too
        for r in Model("track", 0):           # webcam/video -> a stream
            ...
        ai = Model("face_match")              # no input -> a reusable object

    A photo, a sound file, or a piece of text answers with **one** result; a
    webcam index, a video file, or "mic" answers with a stream to iterate; a
    folder answers with a list. ``Model("face_analyze")("group.jpg")`` (build
    first, call later) keeps working — the object form is the same thing.

    Capability names (``face_analyze``, ``read_text``, ``track``, ``gaze``, ...)
    build a :class:`~ovkit.pipelines.base.Pipeline` that chains the models the
    answer needs. They behave exactly like a model: same sources, same
    :class:`~ovkit.Results`. :func:`ovkit.list_pipelines` lists them.

    Parameters
    ----------
    model:
        A registered model name (``"rtdetr_r50"``), a capability name
        (``"face_analyze"``), or a path to an IR ``.xml`` / ``.onnx`` file.
    task:
        Override task auto-detection (``"detect"``/``"classify"``/...).
    device:
        Default OpenVINO device (``"AUTO"``/``"CPU"``/``"GPU"``/``"NPU"``).
        Can be overridden per call.
    precision:
        Target IR precision for conversion (defaults to the manifest value, or
        ``fp16``).
    **kwargs:
        Passed to the pipeline when ``model`` names a capability (for example
        ``Model("face_analyze", attributes=("age_gender",))``).
    """

    #: Options that belong to a *call*, not to building the model/pipeline.
    _RUN_OPTS = ("conf", "imgsz")

    def __new__(cls, model: str | Path = "", source: Any = None, /, **kwargs: Any) -> Any:
        """Dispatch on the name, and run immediately when an input is given.

        Keeping this in ``__new__`` makes ``Model`` the single thing to learn:
        a beginner never has to know whether "face_analyze" is one network or
        four, or that photos and webcams need different plumbing.
        """
        if cls is not Model:
            return super().__new__(cls)
        name: Any = canonical(model) if isinstance(model, str) else model

        if isinstance(name, str):
            from ..pipelines import build_pipeline, is_pipeline

            if is_pipeline(name):
                device = kwargs.pop("device", None) or "AUTO"
                kwargs.pop("task", None)
                kwargs.pop("precision", None)
                run_opts = {k: kwargs.pop(k) for k in cls._RUN_OPTS if k in kwargs}
                pipe = build_pipeline(name, device=device, **kwargs)
                if source is not None:
                    return _immediate(pipe, source, **run_opts)
                return pipe

        if source is not None:
            run_opts = {k: kwargs.pop(k) for k in cls._RUN_OPTS if k in kwargs}
            inst = super().__new__(cls)
            inst.__init__(name, **kwargs)
            return _immediate(inst, source, **run_opts)
        return super().__new__(cls)

    @classmethod
    def network(
        cls,
        model: str | Path,
        task: str | None = None,
        device: str = "AUTO",
        precision: str | None = None,
    ) -> Model:
        """Load one network, even when the name also names a capability.

        ``Model("gaze")`` gives you the composed pipeline, which is what a
        caller wants — but the pipeline itself needs the raw gaze network, and
        so does anyone who wants to drive it by hand. This skips the capability
        dispatch in :meth:`__new__` and always returns a plain model.
        """
        obj = object.__new__(cls)
        obj.__init__(model, task, device, precision)  # type: ignore[misc]
        return obj

    def __init__(
        self,
        model: str | Path,
        source: Any = None,  # consumed by __new__; present here only for the signature
        task: str | None = None,
        device: str = "AUTO",
        precision: str | None = None,
    ) -> None:
        if isinstance(model, str):
            model = canonical(model)
        self.device = device
        self._task_override = task
        self._entry: ModelEntry | None = None
        self._backends: dict[str, Backend] = {}
        self._adapter = None
        self.task: str | None = None
        self.imgsz, self._pre, self._post, self._names = 640, {}, {}, {}
        self.ir_path = self._resolve(model, precision)

    # -- construction helpers ----------------------------------------------

    def _resolve(self, model: str | Path, precision: str | None) -> Path:
        # 1) An existing local file (IR or ONNX) is used directly.
        p = Path(model)
        if p.exists() and p.suffix in {".xml", ".onnx"}:
            self.imgsz = self.imgsz or 640
            return p

        # 2) A registered manifest name.
        entry = resolve(str(model))
        if entry is not None:
            if entry.src == "genai":
                raise OVKitError(
                    f"'{model}' is a genai model — use ovkit.genai.pipeline('{model}') "
                    f"instead of Model() (LLM/STT/TTS/text2image/VLM)."
                )
            self._entry = entry
            self.imgsz = entry.imgsz or 640
            self._pre = entry.preprocess
            self._post = entry.postprocess
            self._names = class_names(entry.postprocess.get("classes"))
            # A descriptor vector and a class-logit vector are numerically
            # indistinguishable, so take the model's own word for it: re-id and
            # retrieval models say so in their name/description.
            haystack = f"{entry.name} {entry.description or ''}".lower()
            if any(k in haystack for k in ("embedding", "re-identification", "reid", "retrieval")):
                self._post = {**self._post, "kind": "embedding"}
            prec = precision or entry.precision
            source = fetch(entry)
            return to_ir(source, entry.name, prec)

        # 3) Unknown. Suggest the closest real name — a typo in Korean or
        # English should cost one glance, not a trip to the docs.
        raise ModelNotFoundError(_unknown_name_message(str(model)))

    def _backend_for(self, device: str) -> Backend:
        if device not in self._backends:
            self._backends[device] = Backend(self.ir_path, device)
        return self._backends[device]

    def _ensure_adapter(self, backend: Backend):
        if self._adapter is None:
            manifest_task = self._entry.task if self._entry else None
            self.task = detect_task(backend, manifest_task, self._task_override)
            from ..recognize.base import BaseAdapter

            # A mirrored model can ship its own class names; without them a
            # classifier can only answer "class_615", which is no answer.
            names = self._names or BaseAdapter.labels_beside(str(self.ir_path))
            self._adapter = get_adapter(
                self.task,
                imgsz=self.imgsz,
                preprocess=self._pre,
                postprocess=self._post,
                names=names or None,
            )
        return self._adapter

    # -- low-level (any model, your own input tensors) ----------------------

    @property
    def inputs(self) -> list[tuple[str, tuple[int, ...], str]]:
        """Return ``(name, shape, dtype)`` for each model input.

        Useful for non-image models (NLP / audio / time series): build matching
        tensors and pass them to :meth:`infer`.
        """
        backend = self._backend_for(self.device)
        info: list[tuple[str, tuple[int, ...], str]] = []
        for inp in backend.compiled.inputs:
            try:
                name = inp.get_any_name()
            except RuntimeError:
                name = ""
            ps = inp.get_partial_shape()
            shape = tuple(int(d.get_length()) if d.is_static else -1 for d in ps)
            info.append((name, shape, str(inp.get_element_type())))
        return info

    def infer(self, inputs: Any, *, device: str | None = None) -> dict[str, np.ndarray]:
        """Run the model on raw input tensor(s), returning ``{name: ndarray}``.

        The escape hatch for any model — including non-image ones (NLP / audio /
        time series) — where you provide the input tensors yourself (see
        :attr:`inputs` for the expected shapes). No image preprocessing is done.
        """
        backend = self._backend_for(device or self.device)
        return backend.infer(inputs)

    # -- prediction ---------------------------------------------------------

    def predict(
        self,
        source: Any,
        *,
        device: str | None = None,
        imgsz: int | None = None,
        conf: float = 0.25,
        stream: bool = False,
        **kwargs: Any,
    ) -> list[Results] | Iterator[Results]:
        """Run prediction on ``source``.

        The input type is auto-detected. An **image** (path / ``ndarray`` /
        folder / video / camera ``int``) runs the vision pipeline. An **audio**
        file on a sound model is read, resampled, framed and decoded into
        :class:`Results` like any other task. Anything else (a ``.npy`` tensor,
        a raw non-image ``ndarray``) is fed to the model directly and the raw
        ``{name: ndarray}`` outputs are returned. ``stream=True`` returns a
        generator for image sources.
        """
        dev = device or self.device

        # Audio in, on a model that takes audio -> a decoded result, not tensors.
        audio = self._maybe_audio_input(source)
        if audio is not None:
            return [self._predict_audio(*audio, device=dev)]

        # Other non-image input -> run the model directly, return raw outputs.
        raw = self._maybe_raw_input(source)
        if raw is not None:
            return self.infer(raw, device=dev)

        backend = self._backend_for(dev)
        # Multi-input models: OK when every input is an image (e.g. super-resolution
        # takes the image + a pre-upscaled copy — the adapter feeds both). Anything
        # else (gaze: eye crops + head-pose angles) can't be driven by one image.
        if len(backend.inputs) > 1 and not _all_image_inputs(backend):
            from ..pipelines import capability_using

            names = ", ".join(n for n, _s, _d in self.inputs)
            # Model has no .name: a model can be a bare path, so identify it by
            # its manifest entry when there is one and by the file otherwise.
            entry_name = self._entry.name if self._entry else Path(self.ir_path).stem
            capability = capability_using(entry_name)
            hint = (
                f"Model({capability!r}) builds those inputs for you."
                if capability
                else "Feed them yourself with model.infer({...}) — see model.inputs."
            )
            raise OVKitError(
                f"{entry_name} needs {len(backend.inputs)} separate inputs ({names}), "
                f"so one image cannot drive it. {hint}"
            )
        adapter = self._ensure_adapter(backend)
        if imgsz is not None:
            adapter.imgsz = imgsz

        gen = self._predict_stream(adapter, backend, source, conf=conf, **kwargs)
        return gen if stream else list(gen)

    def _maybe_raw_input(self, source: Any) -> Any | None:
        """Return a tensor for non-image input (``.npy``/``.wav``/raw array), else None."""
        if isinstance(source, np.ndarray):
            # HWC image (1/3/4 channels) -> vision pipeline; anything else is raw.
            if source.ndim == 3 and source.shape[2] in (1, 3, 4):
                return None
            return source.astype(np.float32)
        if isinstance(source, (str, Path)):
            ext = Path(source).suffix.lower()
            if ext in {".npy"}:
                return np.load(source).astype(np.float32)
            if ext in _AUDIO_EXT:
                return self._load_audio(source)
        return None

    #: Tasks driven by audio rather than an image.
    AUDIO_TASKS = frozenset({"sound_classification", "noise_suppression"})

    def _maybe_audio_input(self, source: Any) -> tuple[np.ndarray, int] | None:
        """Return ``(samples, sample_rate)`` when this is audio for a sound model."""
        task = (self._entry.task if self._entry else None) or self._task_override
        if task not in self.AUDIO_TASKS:
            return None
        if isinstance(source, tuple) and len(source) == 2:  # (samples, sample_rate)
            samples, sr = source
            return np.asarray(samples, np.float32).reshape(-1), int(sr)
        if isinstance(source, (str, Path)) and Path(source).suffix.lower() in _AUDIO_EXT:
            from ..audio import read_audio

            return read_audio(source, target_sr=DEFAULT_AUDIO_SR)
        return None

    def _predict_audio(self, audio: np.ndarray, sr: int, *, device: str) -> Results:
        """Decode an audio clip with the adapter for this model's task."""
        from ..recognize.audio import Denoiser, SoundClassifier

        task = (self._entry.task if self._entry else None) or self._task_override
        if task == "noise_suppression":
            return Denoiser().run(self, audio, sr)
        post = self._entry.postprocess if self._entry else {}
        classifier = SoundClassifier(classes=post.get("classes"), names=self._names or None)
        return classifier.run(self._backend_for(device), audio, sr)

    def _load_audio(self, path: str | Path) -> np.ndarray:
        """Load a ``.wav`` into a float32 tensor shaped to the model's input."""
        import wave

        ext = Path(path).suffix.lower()
        if ext != ".wav":
            raise OVKitError(
                f"Auto-loading '{ext}' audio needs an extra decoder. Convert to .wav, "
                f"or build the input tensor yourself and call model.infer(...)."
            )
        with wave.open(str(path), "rb") as wf:
            n = wf.getnframes()
            raw = wf.readframes(n)
        audio = (np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0)[None]
        # Fit to the model's first static input length when known.
        shape = self.inputs[0][1] if self.inputs else ()
        target = next((d for d in reversed(shape) if d and d > 1), 0)
        if target:
            audio = audio[:, :target]
            if audio.shape[1] < target:
                audio = np.pad(audio, ((0, 0), (0, target - audio.shape[1])))
        return audio

    def __call__(self, source: Any, **kwargs: Any) -> list[Results] | Iterator[Results]:
        """Alias for :meth:`predict` (the model object is callable)."""
        return self.predict(source, **kwargs)

    def _predict_stream(
        self, adapter, backend: Backend, source: Any, *, conf: float, **kwargs: Any
    ) -> Iterator[Results]:
        import time

        for image, path in _iter_sources(source):
            start = time.perf_counter()
            res = adapter.run(backend, image, conf=conf, **kwargs)
            res.elapsed_ms = (time.perf_counter() - start) * 1000.0
            res.device = backend.actual_device
            res.path = path
            yield res

    # -- quantization -------------------------------------------------------

    def quantize(
        self, calib_data: Sequence[Any], preset: str = "int8", subset_size: int = 300
    ) -> Path:
        """Post-training quantize the model with NNCF and cache the INT8 IR.

        ``calib_data`` is an iterable of representative inputs (image paths or
        arrays). Requires the ``ovkit[quant]`` extra. After quantization the
        model serves predictions from the INT8 IR.
        """
        try:
            import nncf
            import openvino as ov
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise OVKitError(
                "Quantization needs NNCF. Install it with: pip install 'ovkit[quant]'."
            ) from exc
        if preset != "int8":
            raise OVKitError(f"Unsupported quantization preset '{preset}'. Use 'int8'.")

        # Build the adapter (for preprocessing) without requiring a device infer.
        backend = self._backend_for(self.device)
        adapter = self._ensure_adapter(backend)

        def _transform(item: Any) -> np.ndarray:
            img, _ = next(iter(_iter_sources(item)))
            return adapter.preprocess_square(img)

        dataset = nncf.Dataset(list(calib_data), _transform)
        model = ov.Core().read_model(str(self.ir_path))
        quantized = nncf.quantize(model, dataset, subset_size=subset_size)

        out = self.ir_path.parent.parent / "int8" / "model.xml"
        out.parent.mkdir(parents=True, exist_ok=True)
        ov.save_model(quantized, str(out))

        self.ir_path = out
        self._backends.clear()  # force recompile from the INT8 IR
        return out

    def __repr__(self) -> str:
        name = self._entry.name if self._entry else self.ir_path.name
        return f"Model(name={name!r}, task={self.task!r}, device={self.device!r})"


# --- source iteration ------------------------------------------------------


def _unknown_name_message(name: str) -> str:
    """A readable, suggestion-bearing message for an unknown model name."""
    import difflib

    from ..pipelines import PIPELINES
    from .i18n import KO_CAPS

    known = sorted(set(list_models()) | set(PIPELINES) | set(KO_CAPS))
    close = difflib.get_close_matches(name, known, n=1, cutoff=0.5)
    hint = ""
    if close:
        target = canonical(close[0])
        shown = close[0] if close[0] == target else f"{close[0]}({target})"
        hint = f" 혹시 '{shown}'?" if lang() == "ko" else f" Did you mean '{shown}'?"
    if lang() == "ko":
        return (
            f"'{name}'은(는) 등록된 모델이나 기능이 아니에요.{hint}\n"
            f"전체 목록: ovkit list 명령이나 list_models() / list_pipelines()"
        )
    return (
        f"'{name}' is not a registered model or capability.{hint}\n"
        f"See `ovkit list`, list_models(), or list_pipelines() for what exists."
    )


def _immediate(runner: Any, source: Any, **run_opts: Any) -> Any:
    """Run ``Model(name, input)`` and shape the answer to the input.

    A photo, a sound file, or an array answers with **one** ``Results`` — a
    student three weeks into Python has not met list indexing yet. A folder
    answers with a list (it visibly holds many). A webcam index, a video file
    or ``"mic"`` answers with the lazy stream ``predict(stream=True)`` builds,
    so nothing opens until iteration starts.
    """
    if isinstance(source, int):
        return runner.predict(source, stream=True, **run_opts)
    if isinstance(source, (str, Path)):
        text = str(source)
        if text == "mic" or Path(text).suffix.lower() in _VIDEO_EXT:
            return runner.predict(source, stream=True, **run_opts)
    out = runner.predict(source, **run_opts)
    if isinstance(out, list):
        if isinstance(source, (str, Path)) and Path(str(source)).is_dir():
            return out
        return out[0] if len(out) == 1 else out
    return out


def _all_image_inputs(backend: Any) -> bool:
    """True when every model input is a static 4-D image ``[N, 1|3, H, W]``."""
    for inp in backend.inputs:
        ps = inp.get_partial_shape()
        dims = [int(d.get_length()) if d.is_static else -1 for d in ps]
        if len(dims) != 4 or dims[1] not in (1, 3) or dims[2] <= 0 or dims[3] <= 0:
            return False
    return True


def _iter_sources(source: Any) -> Iterator[tuple[np.ndarray, str | None]]:
    """Yield ``(image, path)`` pairs from any supported source kind."""
    # Already an array.
    if isinstance(source, np.ndarray):
        yield source, None
        return

    # Camera index.
    if isinstance(source, int):
        yield from _iter_video(source, label=f"camera:{source}")
        return

    # A list/tuple of sources.
    if isinstance(source, (list, tuple)):
        for item in source:
            yield from _iter_sources(item)
        return

    p = Path(str(source))
    if p.is_dir():
        files = sorted(f for f in p.iterdir() if f.suffix.lower() in _IMAGE_EXT)
        if not files:
            raise FileNotFoundError(f"No images found in directory: {p}")
        for f in files:
            yield ops.imread(f), str(f)
        return

    if p.is_file():
        ext = p.suffix.lower()
        if ext in _VIDEO_EXT:
            yield from _iter_video(str(p), label=str(p))
            return
        if ext in _IMAGE_EXT or True:  # try as image; imread raises if invalid
            yield ops.imread(p), str(p)
            return

    raise FileNotFoundError(f"Source not found or unsupported: {source}")


def _iter_video(target: str | int, label: str) -> Iterator[tuple[np.ndarray, str | None]]:
    import cv2

    cap = cv2.VideoCapture(target)
    if not cap.isOpened():
        raise OSError(f"Could not open video source: {label}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame, label
    finally:
        cap.release()
