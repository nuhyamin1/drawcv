"""Custom exception hierarchy for DrawCV."""


class DrawCVError(Exception):
    """Base exception for all DrawCV errors."""
    pass


class ValidationError(DrawCVError, ValueError):
    """Raised when an invalid argument or configuration is passed."""
    pass


class ObjectNotFoundError(DrawCVError, KeyError):
    """Raised when a requested drawable object is not found in a scene."""
    pass


class RenderError(DrawCVError, RuntimeError):
    """Raised when a rendering failure occurs or an unsupported rendering feature is requested."""
    pass
