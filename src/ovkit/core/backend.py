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
        self.compiled = c.compile_model(src, device)
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
