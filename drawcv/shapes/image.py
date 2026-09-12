"""Retained-mode raster image drawable for DrawCV."""

from __future__ import annotations
import base64
import copy
import math
from typing import Any
import cv2
import numpy as np

from drawcv.core.bounds import BoundingBox
from drawcv.core.drawable import Drawable
from drawcv.core.enums import ImageInterpolation
from drawcv.core.exceptions import SerializationError, ValidationError
from drawcv.core.geometry import Point
from drawcv.core.transform import Transform



class ImageObject(Drawable):
    """Retained-mode bitmap image drawable.
    
    Supports BGR and BGRA image matrices, sub-rectangle cropping,
    non-uniform scaling, affine transformations, and interpolation modes.
    """

    def __init__(
        self,
        image: np.ndarray,
        position: Point | tuple[float, float] = Point(0, 0),
        width: float | None = None,
        height: float | None = None,
        crop: BoundingBox | tuple[float, float, float, float] | None = None,
        *,
        id: str | None = None,
        name: str | None = None,
        visible: bool = True,
        locked: bool = False,
        opacity: float = 1.0,
        z_index: int = 0,
        tags: set[str] | None = None,
        metadata: dict[str, Any] | None = None,
        transform: Transform | None = None,
        clip: Any | None = None,
        mask: Any | None = None,
        effects: list[Any] | None = None,
        interpolation: ImageInterpolation = ImageInterpolation.LINEAR,
    ):
        super_kwargs: dict[str, Any] = {
            "name": name,
            "visible": visible,
            "locked": locked,
            "opacity": opacity,
            "z_index": z_index,
            "clip": clip,
            "mask": mask,
        }
        if id is not None:
            super_kwargs["id"] = id
        if tags is not None:
            super_kwargs["tags"] = tags
        if metadata is not None:
            super_kwargs["metadata"] = metadata
        if transform is not None:
            super_kwargs["transform"] = transform
        if effects is not None:
            super_kwargs["effects"] = effects

        super().__init__(**super_kwargs)

        self._validate_image(image)
        self.image = image
        self.position = Point(position[0], position[1]) if isinstance(position, (tuple, list)) else position
        if not isinstance(self.position, Point):
            raise ValidationError(f"position must be a Point, got {type(position).__name__}")

        self.width = float(width) if width is not None else None
        self.height = float(height) if height is not None else None
        if self.width is not None and (self.width <= 0 or math.isnan(self.width) or math.isinf(self.width)):
            raise ValidationError(f"ImageObject width must be positive and finite, got {self.width}")
        if self.height is not None and (self.height <= 0 or math.isnan(self.height) or math.isinf(self.height)):
            raise ValidationError(f"ImageObject height must be positive and finite, got {self.height}")

        self.crop = self._validate_crop(crop)

        if isinstance(interpolation, str):
            try:
                self.interpolation = ImageInterpolation(interpolation.strip().lower())
            except ValueError:
                raise ValidationError(f"Invalid ImageInterpolation '{interpolation}'")
        elif isinstance(interpolation, ImageInterpolation):
            self.interpolation = interpolation
        else:
            raise ValidationError(f"interpolation must be ImageInterpolation, got {type(interpolation).__name__}")

    def _validate_image(self, image: np.ndarray) -> None:
        if not isinstance(image, np.ndarray):
            raise ValidationError(f"Image must be a NumPy array, got {type(image).__name__}")
        if image.dtype != np.uint8:
            raise ValidationError(f"Image dtype must be uint8, got {image.dtype}")
        if image.ndim == 2:
            if image.shape[0] <= 0 or image.shape[1] <= 0:
                raise ValidationError("Image dimensions must be non-empty")
        elif image.ndim == 3:
            if image.shape[0] <= 0 or image.shape[1] <= 0:
                raise ValidationError("Image dimensions must be non-empty")
            if image.shape[2] not in (1, 3, 4):
                raise ValidationError(f"Image channels must be 1, 3 (BGR), or 4 (BGRA), got {image.shape[2]}")
        else:
            raise ValidationError(f"Image array must be 2D or 3D, got {image.ndim}D")

    def _validate_crop(self, crop: BoundingBox | tuple[float, float, float, float] | None) -> BoundingBox | None:
        if crop is None:
            return None
        if isinstance(crop, (tuple, list)):
            if len(crop) != 4:
                raise ValidationError("Crop tuple must have 4 elements: (x, y, width, height)")
            crop_box = BoundingBox(crop[0], crop[1], crop[2], crop[3])
        elif isinstance(crop, BoundingBox):
            crop_box = crop
        else:
            raise ValidationError(f"Crop must be a BoundingBox or 4-tuple, got {type(crop).__name__}")

        img_h, img_w = self.image.shape[:2]
        if crop_box.width <= 0 or crop_box.height <= 0:
            raise ValidationError(f"Crop dimensions must be positive, got ({crop_box.width}, {crop_box.height})")
        if crop_box.left < 0 or crop_box.top < 0 or crop_box.right > img_w or crop_box.bottom > img_h:
            raise ValidationError(
                f"Crop box ({crop_box.left}, {crop_box.top}, {crop_box.width}, {crop_box.height}) "
                f"exceeds image dimensions ({img_w}, {img_h})"
            )
        return crop_box

    @property
    def source_width(self) -> int:
        """Native width of the source image matrix in pixels."""
        return int(self.image.shape[1])

    @property
    def source_height(self) -> int:
        """Native height of the source image matrix in pixels."""
        return int(self.image.shape[0])

    @property
    def display_width(self) -> float:
        """Rendered display width (accounting for crop and custom width)."""
        if self.width is not None:
            return self.width
        if self.crop is not None:
            return self.crop.width
        return float(self.source_width)

    @property
    def display_height(self) -> float:
        """Rendered display height (accounting for crop and custom height)."""
        if self.height is not None:
            return self.height
        if self.crop is not None:
            return self.crop.height
        return float(self.source_height)

    @property
    def channels(self) -> int:
        """Number of color channels in source image (1, 3, or 4)."""
        return 1 if self.image.ndim == 2 else int(self.image.shape[2])

    # -------------------------------------------------------------------------
    # Bounds & Geometry
    # -------------------------------------------------------------------------

    def get_geometry_bounds(self) -> BoundingBox:
        """Intrinsic geometric bounds in local coordinates."""
        return BoundingBox(self.position.x, self.position.y, self.display_width, self.display_height)

    def get_local_bounds(self) -> BoundingBox:
        """Visual bounds at identity transform."""
        return self.get_geometry_bounds()

    def get_bounds(self) -> BoundingBox:
        """World-space bounding box enclosing transformed image corners."""
        gx, gy = self.position.x, self.position.y
        gw, gh = self.display_width, self.display_height
        corners = [
            Point(gx, gy),
            Point(gx + gw, gy),
            Point(gx + gw, gy + gh),
            Point(gx, gy + gh),
        ]
        world_pts = [self.to_world(p) for p in corners]
        xs = [p.x for p in world_pts]
        ys = [p.y for p in world_pts]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        return BoundingBox(min_x, min_y, max_x - min_x, max_y - min_y)

    # -------------------------------------------------------------------------
    # Anchors, Hit-Testing & Cloning
    # -------------------------------------------------------------------------

    def anchor(self, name: str) -> Point:
        """Retrieve a named geometric anchor point in world coordinates."""
        norm = name.strip().lower().replace("-", "_").replace(" ", "_")
        b = self.get_bounds()

        if norm == "center":
            return b.center
        elif norm == "top":
            return Point(b.left + b.width / 2.0, b.top)
        elif norm == "bottom":
            return Point(b.left + b.width / 2.0, b.bottom)
        elif norm == "left":
            return Point(b.left, b.top + b.height / 2.0)
        elif norm == "right":
            return Point(b.right, b.top + b.height / 2.0)
        elif norm in ("top_left", "topleft"):
            return b.top_left
        elif norm in ("top_right", "topright"):
            return b.top_right
        elif norm in ("bottom_left", "bottomleft"):
            return b.bottom_left
        elif norm in ("bottom_right", "bottomright"):
            return b.bottom_right
        else:
            valid = "center, top, bottom, left, right, top_left, top_right, bottom_left, bottom_right"
            raise ValidationError(f"Unknown anchor '{name}' for ImageObject. Valid anchors are: {valid}")

    def contains_point(self, world_point: Point) -> bool:
        """Test whether a world-space point intersects the transformed image rectangle."""
        if not self.effective_visible:
            return False
        local_pt = self.to_local(world_point)
        gx, gy = self.position.x, self.position.y
        gw, gh = self.display_width, self.display_height
        return (gx <= local_pt.x <= gx + gw) and (gy <= local_pt.y <= gy + gh)

    def clone(self, new_id: bool = True) -> ImageObject:
        """Create a deep copy of this ImageObject."""
        import uuid
        return ImageObject(
            image=self.image.copy(),
            position=Point(self.position.x, self.position.y),
            width=self.width,
            height=self.height,
            crop=copy.deepcopy(self.crop),
            id=str(uuid.uuid4()) if new_id else self.id,
            name=self.name,
            visible=self.visible,
            locked=self.locked,
            opacity=self.opacity,
            z_index=self.z_index,
            tags=set(self.tags),
            metadata=copy.deepcopy(self.metadata),
            transform=copy.deepcopy(self.transform),
            clip=copy.deepcopy(self.clip),
            mask=copy.deepcopy(self.mask),
            effects=copy.deepcopy(self.effects),
            interpolation=self.interpolation,
        )

    def _get_shape_state(self) -> dict[str, Any]:
        return {
            "image": self.image.copy(),
            "position": self.position.copy(),
            "width": float(self.width) if self.width is not None else None,
            "height": float(self.height) if self.height is not None else None,
            "crop": copy.deepcopy(self.crop),
            "interpolation": self.interpolation.value if hasattr(self.interpolation, "value") else str(self.interpolation),
        }

    def _apply_shape_state(self, state: dict[str, Any]) -> None:
        if "image" in state:
            self.image = state["image"].copy()
        if "position" in state:
            self.position = state["position"].copy() if isinstance(state["position"], Point) else Point.from_dict(state["position"])
        if "width" in state:
            self.width = float(state["width"]) if state["width"] is not None else None
        if "height" in state:
            self.height = float(state["height"]) if state["height"] is not None else None
        if "crop" in state:
            c = state["crop"]
            self.crop = copy.deepcopy(c) if isinstance(c, BoundingBox) else (BoundingBox.from_dict(c) if c is not None else None)
        if "interpolation" in state:
            it = state["interpolation"]
            self.interpolation = ImageInterpolation(it) if isinstance(it, str) else it

    def to_dict(self) -> dict[str, Any]:
        """Return a plain JSON-compatible dictionary representation with structured raster payload."""
        res = self._base_to_dict()
        success, encoded = cv2.imencode(".png", self.image)
        if not success:
            raise SerializationError("Failed to encode ImageObject buffer to PNG")
        data_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")

        res.update({
            "type": "image",
            "image": {
                "encoding": "png_base64",
                "dtype": "uint8",
                "shape": list(self.image.shape),
                "data": data_b64,
            },
            "position": self.position.to_dict(),
            "width": float(self.width) if self.width is not None else None,
            "height": float(self.height) if self.height is not None else None,
            "crop": self.crop.to_dict() if self.crop is not None else None,
            "interpolation": self.interpolation.value if hasattr(self.interpolation, "value") else str(self.interpolation),
        })
        return res

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ImageObject:
        """Construct an ImageObject from dictionary representation."""
        base_kwargs = cls._base_from_dict(data)
        img_payload = data.get("image")
        if not isinstance(img_payload, dict):
            raise SerializationError("ImageObject 'image' must be a structured raster dictionary")
        if img_payload.get("encoding") != "png_base64":
            raise SerializationError(f"Unsupported image encoding '{img_payload.get('encoding')}'")

        raw_bytes = base64.b64decode(img_payload["data"])
        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        decoded = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise SerializationError("Failed to decode ImageObject PNG buffer")

        expected_shape = tuple(img_payload.get("shape", []))
        if expected_shape and decoded.shape != expected_shape:
            raise SerializationError(
                f"Decoded image shape {decoded.shape} does not match expected shape {expected_shape}"
            )

        pos = Point.from_dict(data["position"]) if "position" in data else Point(0.0, 0.0)
        crop_val = BoundingBox.from_dict(data["crop"]) if data.get("crop") is not None else None
        interp = ImageInterpolation(data["interpolation"]) if "interpolation" in data else ImageInterpolation.LINEAR

        return cls(
            image=decoded,
            position=pos,
            width=float(data["width"]) if data.get("width") is not None else None,
            height=float(data["height"]) if data.get("height") is not None else None,
            crop=crop_val,
            interpolation=interp,
            **base_kwargs,
        )

