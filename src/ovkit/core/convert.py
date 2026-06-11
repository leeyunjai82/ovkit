"""Convert source models (ONNX / IR) to OpenVINO IR, with a conversion cache.

Conversion runs at most once per ``(name, precision)``: the resulting IR is
written to the model cache and reused on subsequent loads.
"""

from __future__ import annotations

from pathlib import Path

from .download import model_cache_dir
from .errors import ConversionError


def _ir_paths(name: str, precision: str) -> tuple[Path, Path]:
    base = model_cache_dir(name) / "ir" / precision
    return base / "model.xml", base / "model.bin"


def cached_ir(name: str, precision: str) -> Path | None:
    """Return the cached IR ``.xml`` for ``(name, precision)`` if it exists."""
    xml, _ = _ir_paths(name, precision)
    return xml if xml.is_file() else None


def to_ir(source: Path, name: str, precision: str = "fp16") -> Path:
    """Convert ``source`` to OpenVINO IR and return the cached ``.xml`` path.

    ``source`` may already be IR (``.xml``) — in that case it is passed through
    unchanged. ONNX sources are converted with ``openvino.convert_model`` and
    serialized (compressing weights to fp16 when ``precision == "fp16"``).
    The result is cached so conversion happens only once.
    """
    source = Path(source)

    # Already IR: load directly, no conversion/caching needed.
    if source.suffix == ".xml":
        return source

    cached = cached_ir(name, precision)
    if cached is not None:
        return cached

    try:
        import openvino as ov
    except ImportError as exc:  # pragma: no cover - dependency missing
        raise ConversionError("openvino is required to convert models to IR.") from exc

    if source.suffix not in {".onnx", ".pb", ".pdmodel"} and not source.is_file():
        raise ConversionError(f"Cannot convert '{source}': unsupported or missing source.")

    xml_path, _ = _ir_paths(name, precision)
    xml_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        ov_model = ov.convert_model(str(source))
        compress = precision == "fp16"
        ov.save_model(ov_model, str(xml_path), compress_to_fp16=compress)
    except Exception as exc:  # pragma: no cover - conversion variety
        raise ConversionError(f"Failed to convert '{source.name}' to IR: {exc}") from exc

    if not xml_path.is_file():  # pragma: no cover - defensive
        raise ConversionError(f"Conversion produced no IR at {xml_path}.")
    return xml_path
