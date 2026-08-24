"""Thin OpenVINO runtime wrapper: device abstraction, sync + async inference.

A :class:`Backend` owns a compiled model for a chosen device and exposes both a
single-shot :meth:`infer` (synchronous) and a throughput-oriented
:meth:`infer_batch` built on ``ov.AsyncInferQueue`` for streams/folders/video.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

#: A single OpenVINO Core, shared process-wide (creating many is wasteful).
_CORE = None


def core() -> Any:
    """Return the shared :class:`openvino.Core`, creating it on first use."""
    global _CORE
    if _CORE is None:
        import openvino as ov

        _CORE = ov.Core()
    return _CORE


def available_devices() -> list[str]:
    """Return device names visible to OpenVINO (e.g. ``["CPU", "GPU", "NPU"]``)."""
    return list(core().available_devices)


def _friendlier_compile_error(exc: Exception, model: str | Path | Any) -> Exception:
    """Turn OpenVINO's nested compile errors into something actionable.

    The most common one by far is IR whose ``.bin`` never arrived (a mirror
    upload that dropped the weights, or an interrupted download). OpenVINO
    reports it as "Empty weights data in bin file", buried under two layers of
    "Exception from ...", which tells a user nothing about what to do.
    """
    from .errors import OVKitError

    text = str(exc)
    if "Empty weights data" not in text and "bin file" not in text:
        return exc
    if not isinstance(model, (str, Path)):
        return exc
    xml = Path(str(model))
    if xml.suffix != ".xml":
        return exc
    bin_path = xml.with_suffix(".bin")
    if not bin_path.exists():
        state = "is missing"
    elif bin_path.stat().st_size < 1024:
        state = f"is only {bin_path.stat().st_size} bytes"
    else:
        return exc  # weights look fine — a different problem, keep the original
    return OVKitError(
        f"The weights file for this model {state}: {bin_path}\n"
        f"An OpenVINO IR needs both model.xml and model.bin. Delete the cached "
        f"copy and download it again; if it keeps happening, the mirrored model "
        f"itself is incomplete and needs re-uploading."
    )


class Backend:
    """A compiled model bound to a device, with sync and async inference.

    Parameters
    ----------
    model:
        Path to an IR ``.xml`` / ONNX file, or an already-built ``ov.Model``.
    device:
        OpenVINO device string. ``"AUTO"`` (default) lets OpenVINO pick.
    """

    def __init__(self, model: str | Path | Any, device: str = "AUTO") -> None:
        self.device = device
        c = core()
        src = str(model) if isinstance(model, (str, Path)) else model
        try:
            self.compiled = c.compile_model(src, device)
        except Exception as exc:
            raise _friendlier_compile_error(exc, model) from exc
        self.inputs = self.compiled.inputs
        self.outputs = self.compiled.outputs

    # -- introspection ------------------------------------------------------

    @property
    def input_shape(self) -> tuple[int, ...]:
        """Partial shape of the first input as a tuple (``-1`` for dynamic)."""
        ps = self.compiled.inputs[0].get_partial_shape()
        dims: list[int] = []
        for d in ps:
            dims.append(int(d.get_length()) if d.is_static else -1)
        return tuple(dims)

    def _adapt_image_channels(self, arr: np.ndarray) -> np.ndarray:
        """Match a single 4-D image tensor to the model's layout and channels.

        The preprocessor always produces NCHW with 3 channels; models differ:
        TF-converted OMZ models take **NHWC** (channels last), and OCR /
        grayscale classifiers take **1 channel**. Reconcile both here — every
        image adapter funnels through :meth:`infer`. Anything that doesn't look
        like a single 4-D image passes through unchanged.
        """
        shape = self.input_shape
        if arr.ndim != 4 or len(shape) != 4:
            return arr

        # NHWC model (channels last, e.g. [1, 224, 224, 3]): transpose our NCHW.
        if shape[-1] in (1, 3) and shape[1] not in (1, 3):
            if arr.shape[1] in (1, 3) and arr.shape[-1] not in (1, 3):
                arr = np.transpose(arr, (0, 2, 3, 1))
            exp, got = shape[-1], arr.shape[-1]
            if exp == 1 and got == 3:
                arr = arr.mean(axis=3, keepdims=True).astype(arr.dtype)
            elif exp == 3 and got == 1:
                arr = np.repeat(arr, 3, axis=3)
            return np.ascontiguousarray(arr)

        if shape[1] not in (1, 3):
            return arr
        exp, got = shape[1], arr.shape[1]
        if got == exp:
            return arr
        if exp == 1 and got == 3:  # to grayscale (channel-order agnostic)
            return arr.mean(axis=1, keepdims=True).astype(arr.dtype)
        if exp == 3 and got == 1:  # broadcast gray -> 3 channels
            return np.repeat(arr, 3, axis=1)
        return arr

    @property
    def actual_device(self) -> str:
        """The device inference actually runs on (AUTO resolves to a real one)."""
        try:
            devices = self.compiled.get_property("EXECUTION_DEVICES")
            return ",".join(devices) if devices else self.device
        except Exception:
            return self.device

    def output_signatures(self) -> list[tuple[str, tuple[int, ...]]]:
        """Return ``(name, shape)`` for each output (``-1`` for dynamic dims)."""
        sigs: list[tuple[str, tuple[int, ...]]] = []
        for out in self.compiled.outputs:
            ps = out.get_partial_shape()
            shape = tuple(int(d.get_length()) if d.is_static else -1 for d in ps)
            try:
                name = out.get_any_name()
            except RuntimeError:
                name = ""
            sigs.append((name, shape))
        return sigs

    def rt_info(self, *keys: str) -> str | None:
        """Read a runtime-info value from the underlying model, or ``None``."""
        try:
            model = self.compiled.get_runtime_model()
            info = model.get_rt_info(list(keys))
            return str(info)
        except Exception:
            return None

    # -- inference ----------------------------------------------------------

    def infer(self, inputs: np.ndarray | dict[Any, np.ndarray]) -> dict[str, np.ndarray]:
        """Run one synchronous inference and return ``{output_name: ndarray}``."""
        if isinstance(inputs, np.ndarray):
            inputs = self._adapt_image_channels(inputs)
        result = self.compiled(inputs)
        return self._named(result)

    def infer_batch(
        self,
        feeds: Iterable[np.ndarray | dict[Any, np.ndarray]],
        callback: Callable[[int, dict[str, np.ndarray]], None] | None = None,
        jobs: int = 0,
    ) -> Iterator[dict[str, np.ndarray]]:
        """Run inference over ``feeds`` using an async queue (throughput mode).

        Yields result dicts in completion order. When ``callback`` is given it
        is invoked as ``callback(index, result)``; otherwise results are
        collected and yielded. ``jobs`` sets the number of in-flight requests
        (``0`` lets OpenVINO choose the optimal number).
        """
        import openvino as ov

        queue = ov.AsyncInferQueue(self.compiled, jobs)
        collected: dict[int, dict[str, np.ndarray]] = {}

        def _on_done(request: Any, userdata: int) -> None:
            named = self._named({out: request.get_tensor(out).data for out in self.outputs})
            if callback is not None:
                callback(userdata, named)
            else:
                collected[userdata] = named

        queue.set_callback(_on_done)
        count = 0
        for i, feed in enumerate(feeds):
            if isinstance(feed, np.ndarray):
                feed = self._adapt_image_channels(feed)
            queue.start_async(feed, userdata=i)
            count += 1
        queue.wait_all()

        if callback is None:
            for i in range(count):
                if i in collected:
                    yield collected[i]

    def _named(self, result: Any) -> dict[str, np.ndarray]:
        named: dict[str, np.ndarray] = {}
        for idx, out in enumerate(self.compiled.outputs):
            try:
                name = out.get_any_name()
            except RuntimeError:
                name = f"output_{idx}"
            try:
                named[name] = np.asarray(result[out])
            except (KeyError, TypeError):
                named[name] = np.asarray(result[idx])
        return named
