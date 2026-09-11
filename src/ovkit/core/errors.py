"""Exception hierarchy for ovkit.

All errors raised by ovkit derive from :class:`OVKitError` so callers can catch
the whole family with a single ``except``.
"""

from __future__ import annotations


class OVKitError(Exception):
    """Base class for every error raised by ovkit."""


class ModelNotFoundError(OVKitError):
    """A model name could not be resolved to a local path or manifest entry."""


class OfflineError(OVKitError):
    """Network access was required but ``OVKIT_OFFLINE=1`` is set."""


class DownloadError(OVKitError):
    """A model artifact failed to download or failed its integrity check."""


class GatedModelError(DownloadError):
    """A Hugging Face repo is gated and requires authentication."""


class MirrorMissingError(DownloadError):
    """A model is expected on the ovkit HF mirror but is not (yet) there."""


class ConversionError(OVKitError):
    """Conversion of a source model (ONNX/torch) to OpenVINO IR failed."""


class TaskDetectionError(OVKitError):
    """The task (detect/classify/segment/pose) could not be determined."""


class LicenseError(OVKitError):
    """A model carries a non-permissive license and may not be registered."""


def multi_input_message(model_name: str, input_names: list[str]) -> str:
    """The message for a model fed one image when it needs several inputs.

    Written once because it was drifting into three near-identical copies, and
    because the useful half — naming the capability that builds those inputs —
    only existed in one of them.
    """
    from ..pipelines import capability_using

    capability = capability_using(model_name)
    hint = (
        f"Model({capability!r}) builds those inputs for you."
        if capability
        else "Feed them yourself with model.infer({...}) — see model.inputs."
    )
    return (
        f"{model_name} needs {len(input_names)} separate inputs "
        f"({', '.join(input_names)}), so one image cannot drive it. {hint}"
    )
