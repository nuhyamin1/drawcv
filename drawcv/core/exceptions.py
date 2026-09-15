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


class SerializationError(DrawCVError):
    """Base exception for serialization and deserialization failures."""
    pass


class InvalidFormatError(SerializationError, ValueError):
    """Raised when an unrecognized or malformed document format/envelope is encountered."""
    pass


class UnsupportedVersionError(SerializationError, ValueError):
    """Raised when an unsupported document schema version is encountered."""
    pass


class UnknownDrawableTypeError(SerializationError, KeyError):
    """Raised when an unrecognized drawable type identifier is encountered in serialized data."""
    pass


class PathBooleanError(DrawCVError):
    """Raised when a path boolean operation fails in the geometry backend."""
    pass


