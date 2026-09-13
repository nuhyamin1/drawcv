"""Immutable, content-addressed font bytes for portable retained scenes."""
from dataclasses import dataclass, field
from pathlib import Path
import base64
import hashlib
from drawcv.core.exceptions import ValidationError


@dataclass(frozen=True)
class FontAsset:
    name: str
    data: bytes = field(repr=False)

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValidationError("FontAsset name must be a nonempty string")
        if not isinstance(self.data, bytes) or self.data[:4] not in (b'\x00\x01\x00\x00', b'OTTO'):
            raise ValidationError("FontAsset requires a standalone TrueType or OpenType font")

    @property
    def sha256(self):
        return hashlib.sha256(self.data).hexdigest()

    @classmethod
    def from_file(cls, path):
        path = Path(path)
        try:
            return cls(path.name, path.read_bytes())
        except OSError as exc:
            raise ValidationError(f"Cannot read font asset: {path}") from exc

    def copy(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def to_dict(self):
        return {"name": self.name, "sha256": self.sha256,
                "data": base64.b64encode(self.data).decode("ascii")}

    @classmethod
    def from_dict(cls, data):
        try:
            asset = cls(data["name"], base64.b64decode(data["data"], validate=True))
            if asset.sha256 != data["sha256"]:
                raise ValueError("digest mismatch")
            return asset
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationError("Invalid embedded font asset or SHA-256 mismatch") from exc
