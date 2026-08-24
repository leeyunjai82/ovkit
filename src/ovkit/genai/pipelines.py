"""openvino-genai pipelines, made usable: resolve a name, download, build.

``pipeline(name)`` resolves a registered genai model (``manifests/genai.yaml``),
downloads the OpenVINO-converted model from Hugging Face, and returns the ready
``openvino_genai`` pipeline. A local directory path also works.

    from ovkit.genai import pipeline
    llm = pipeline("tinyllama_chat")
    print(llm.generate("Hello", max_new_tokens=50))

    stt = pipeline("whisper_base")
    print(stt.generate(audio))           # audio: float32 16 kHz mono ndarray

Requires ``pip install -e ".[genai]"`` (openvino-genai + optimum-intel).

The thin per-type factories (``llm_pipeline`` etc.) remain for building a
pipeline directly from a model directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..core.errors import OVKitError
from ..core.registry import resolve

_DEFAULT_DEVICE = "AUTO"


def _require_genai() -> Any:
    try:
        import openvino_genai as ovg
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "openvino-genai is required for LLM / speech / vision-language models.\n"
            'Install it with:  pip install "ovkit[genai]"\n'
            "(checked before downloading, so nothing has been fetched yet)."
        ) from exc
    return ovg


def _builder(ptype: str):
    ovg = _require_genai()
    builders = {
        "llm": ovg.LLMPipeline,
        "whisper": ovg.WhisperPipeline,
        "text2image": ovg.Text2ImagePipeline,
        "text2speech": ovg.Text2SpeechPipeline,
        "vlm": ovg.VLMPipeline,
    }
    if ptype not in builders:
        raise OVKitError(f"Unknown genai pipeline type {ptype!r}. Known: {sorted(builders)}.")
    return builders[ptype]


def _download(entry: Any) -> str:
    """Download a genai model directory: ovkit mirror first, upstream fallback.

    Tries the ovkit mirror (``entry.repo`` + optional ``entry.subfolder``); if
    that repo/subfolder isn't populated yet, falls back to
    ``entry.extra['upstream']`` (the original OpenVINO repo). Returns the local
    model directory.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - dependency missing
        raise OVKitError("huggingface_hub is required to download genai models.") from exc

    # (repo, subfolder) candidates in priority order: mirror, then upstream.
    candidates: list[tuple[str, str | None]] = []
    if entry.repo:
        candidates.append((entry.repo, entry.subfolder))
    upstream = entry.extra.get("upstream")
    if upstream and upstream != entry.repo:
        candidates.append((upstream, None))

    last_err: Exception | None = None
    for repo_id, subfolder in candidates:
        try:
            patterns = [f"{subfolder}/**", f"{subfolder}/*"] if subfolder else None
            local = Path(snapshot_download(repo_id=repo_id, allow_patterns=patterns))
            model_dir = local / subfolder if subfolder else local
            if model_dir.is_dir() and any(model_dir.iterdir()):
                return str(model_dir)
        except Exception as exc:  # try the next candidate (e.g. mirror not populated)
            last_err = exc
    raise OVKitError(
        "Could not download genai model from mirror or upstream "
        f"({[c[0] for c in candidates]})." + (f" Last error: {last_err}" if last_err else "")
    )


def pipeline(name: str, device: str = _DEFAULT_DEVICE, **kwargs: Any) -> Any:
    """Build an openvino-genai pipeline from a registered name or a local path.

    Registered genai names (see ``manifests/genai.yaml``) are downloaded from
    Hugging Face; a local directory is used as-is (its pipeline type is then
    required via ``pipeline_type=...``).
    """
    # Fail before downloading, not after: a genai model is gigabytes, and
    # discovering the missing dependency at build time wasted the whole
    # download.
    _require_genai()

    entry = resolve(name)
    if entry is not None and entry.src == "genai":
        ptype = entry.extra.get("pipeline") or kwargs.pop("pipeline_type", None)
        if not ptype:
            raise OVKitError(f"genai model '{name}' is missing a 'pipeline' type.")
        model_dir = _download(entry) if (entry.repo or entry.extra.get("upstream")) else name
    else:
        model_dir = str(name)
        ptype = kwargs.pop("pipeline_type", None)
        if not ptype:
            raise OVKitError(
                f"'{name}' is not a registered genai model. For a local path, pass "
                f"pipeline_type='llm'|'whisper'|'text2image'|'text2speech'|'vlm'."
            )
        if not Path(model_dir).exists():
            raise OVKitError(f"genai model path not found: {model_dir}")
    return _builder(ptype)(model_dir, device, **kwargs)


# --- thin per-type factories (build directly from a model directory) -------


def llm_pipeline(model_path: str, device: str = _DEFAULT_DEVICE, **kwargs: Any) -> Any:
    """Return an ``openvino_genai.LLMPipeline``."""
    return _require_genai().LLMPipeline(model_path, device, **kwargs)


def text2image_pipeline(model_path: str, device: str = _DEFAULT_DEVICE, **kwargs: Any) -> Any:
    """Return an ``openvino_genai.Text2ImagePipeline``."""
    return _require_genai().Text2ImagePipeline(model_path, device, **kwargs)


def whisper_pipeline(model_path: str, device: str = _DEFAULT_DEVICE, **kwargs: Any) -> Any:
    """Return an ``openvino_genai.WhisperPipeline`` (speech-to-text)."""
    return _require_genai().WhisperPipeline(model_path, device, **kwargs)


def vlm_pipeline(model_path: str, device: str = _DEFAULT_DEVICE, **kwargs: Any) -> Any:
    """Return an ``openvino_genai.VLMPipeline``."""
    return _require_genai().VLMPipeline(model_path, device, **kwargs)


def text2speech_pipeline(model_path: str, device: str = _DEFAULT_DEVICE, **kwargs: Any) -> Any:
    """Return an ``openvino_genai.Text2SpeechPipeline`` (text-to-speech)."""
    return _require_genai().Text2SpeechPipeline(model_path, device, **kwargs)


# --- one-call helpers (the usable surface over the pipelines) --------------
#
# Building a pipeline is only half the job: a VLM wants an ``ov.Tensor`` of
# uint8 RGB, Whisper wants float32 mono at 16 kHz, and an LLM wants its
# generation config. These three functions do that conversion so calling code
# passes an image path or a wav path and gets a string back.

_PIPES: dict[tuple[str, str], Any] = {}


def _cached(name: str, device: str) -> Any:
    """Build a pipeline once per (name, device) — they are expensive to load."""
    key = (name, device)
    if key not in _PIPES:
        _PIPES[key] = pipeline(name, device)
    return _PIPES[key]


def generate(prompt: str, model: str = "llm", device: str = _DEFAULT_DEVICE, **kwargs: Any) -> str:
    """Answer ``prompt`` with a text LLM and return the text.

    >>> from ovkit.genai import generate
    >>> generate("Explain OpenVINO in one sentence.")
    """
    kwargs.setdefault("max_new_tokens", 200)
    return str(_cached(model, device).generate(prompt, **kwargs))


def describe(
    image: Any,
    prompt: str = "Describe this image.",
    model: str = "vlm",
    device: str = _DEFAULT_DEVICE,
    **kwargs: Any,
) -> str:
    """Ask a vision-language model about ``image`` and return its answer.

    ``image`` is a path or a BGR ``ndarray`` (what OpenCV and ovkit hand you);
    it is converted to the RGB tensor the pipeline expects.

        >>> from ovkit.genai import describe
        >>> describe("desk.jpg", "How many monitors are there?")
    """
    tensor = _image_tensor(image)
    pipe = _cached(model, device)
    kwargs.setdefault("max_new_tokens", 200)
    # The images= keyword replaced image= in openvino-genai 2025.x; accept both
    # so ovkit works across the versions people actually have installed.
    try:
        return str(pipe.generate(prompt, images=[tensor], **kwargs))
    except TypeError:
        return str(pipe.generate(prompt, image=tensor, **kwargs))


def transcribe(
    audio: Any,
    model: str = "stt",
    device: str = _DEFAULT_DEVICE,
    **kwargs: Any,
) -> str:
    """Transcribe speech and return the text.

    ``audio`` is a path (any format the audio helpers can read) or a float32
    mono ``ndarray``; it is resampled to the 16 kHz Whisper expects.

        >>> from ovkit.genai import transcribe
        >>> transcribe("meeting.wav")
    """
    import numpy as np

    if isinstance(audio, (str, Path)):
        from ..audio import read_audio

        samples, _sr = read_audio(audio, target_sr=16_000)
    else:
        samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    result = _cached(model, device).generate(samples, **kwargs)
    return str(result).strip()


def _image_tensor(image: Any) -> Any:
    """Convert a path / BGR ndarray into the uint8 RGB ``ov.Tensor`` a VLM wants."""
    import numpy as np
    import openvino as ov

    if isinstance(image, (str, Path)):
        from ..image.ops import imread

        arr = imread(image)
    else:
        arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise OVKitError(f"Expected an HxWx3 image for a vision-language model, got {arr.shape}.")
    arr = arr[:, :, :3][:, :, ::-1]  # BGR -> RGB
    return ov.Tensor(np.ascontiguousarray(arr, dtype=np.uint8)[None])
