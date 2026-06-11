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
