"""Video and multi-frame animation rendering engine for DrawCV."""

from __future__ import annotations
import math
from pathlib import Path
from typing import Generator
import cv2

from drawcv.canvas import Canvas
from drawcv.core.exceptions import RenderError, ValidationError
from drawcv.scene import Scene


class VideoRenderer:
    """Renders temporal scenes to frame generators, image sequences, and video files."""

    @staticmethod
    def render_frames(
        scene: Scene,
        duration: float | None = None,
        fps: int = 30,
        *,
        alpha: bool = False,
    ) -> Generator[Canvas, None, None]:
        """Generate BGR Canvas frames (or straight BGRA with alpha=True).

        Frame count: N = math.ceil(duration * fps) for duration > 0.
        Timestamps: t_k = k / fps for k in [0, N-1].
        """
        if not isinstance(scene, Scene):
            raise ValidationError(f"Expected Scene, got {type(scene).__name__}")
        if not isinstance(alpha, bool):
            raise ValidationError("alpha must be a boolean")
        if not isinstance(fps, int) or isinstance(fps, bool) or fps <= 0:
            raise ValidationError(f"fps must be a positive integer, got {fps}")

        if duration is None:
            effective_duration = scene.temporal_duration
            if math.isinf(effective_duration):
                raise ValidationError(
                    "Scene contains infinite looping animations; explicit duration must be provided to render_frames"
                )
        else:
            if not isinstance(duration, (int, float)) or isinstance(duration, bool):
                raise ValidationError(f"duration must be numeric, got {type(duration).__name__}")
            effective_duration = float(duration)
            if effective_duration < 0.0:
                raise ValidationError(f"duration must be non-negative, got {duration}")

        if effective_duration <= 0.0:
            return

        num_frames = math.ceil(effective_duration * fps)
        for k in range(num_frames):
            time_k = k / float(fps)
            yield scene.render_at_time(time_k, alpha=True) if alpha else scene.render_at_time(time_k)

    @classmethod
    def render_image_sequence(
        cls,
        scene: Scene,
        output_dir: str | Path,
        pattern: str = "frame_%04d.png",
        duration: float | None = None,
        fps: int = 30,
        *,
        alpha: bool = False,
    ) -> list[Path]:
        """Render numbered images; alpha=True requires a PNG filename pattern."""
        if not isinstance(alpha, bool):
            raise ValidationError("alpha must be a boolean")
        if alpha and Path(pattern).suffix.lower() != ".png":
            raise ValidationError("Alpha image sequences require a PNG filename pattern")
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        paths: list[Path] = []
        for idx, canvas in enumerate(cls.render_frames(scene, duration=duration, fps=fps, alpha=alpha)):
            frame_filename = pattern % idx
            frame_path = out_dir / frame_filename
            canvas.save(frame_path)
            paths.append(frame_path)

        return paths

    @classmethod
    def render_video(
        cls,
        scene: Scene,
        output_path: str | Path,
        duration: float | None = None,
        fps: int = 30,
        fourcc: str = "mp4v",
    ) -> Path:
        """Render scene into an encoded video file (MP4/AVI) using OpenCV VideoWriter."""
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
        writer = cv2.VideoWriter(
            str(out_path),
            fourcc_code,
            float(fps),
            (scene.width, scene.height),
        )

        if not writer.isOpened():
            raise RenderError(
                f"OpenCV VideoWriter failed to initialize for '{output_path}' with fourcc '{fourcc}'"
            )

        try:
            for canvas in cls.render_frames(scene, duration=duration, fps=fps):
                writer.write(canvas.buffer)
        finally:
            writer.release()

        return out_path

    @classmethod
    def render_gif(
        cls,
        scene: Scene,
        output_path: str | Path,
        duration: float | None = None,
        fps: int = 20,
        loop: int = 0,
    ) -> Path:
        """Render scene into an animated GIF using Pillow (optional convenience)."""
        try:
            from PIL import Image
        except ImportError:
            raise RenderError("Pillow is required for GIF export. Install Pillow to use render_gif.")

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        pil_frames: list[Any] = []
        for canvas in cls.render_frames(scene, duration=duration, fps=fps):
            # Canvas buffer is BGR, Pillow expects RGB
            rgb_buffer = cv2.cvtColor(canvas.buffer, cv2.COLOR_BGR2RGB)
            pil_frames.append(Image.fromarray(rgb_buffer))

        if not pil_frames:
            raise RenderError("No frames generated for GIF export")

        frame_duration_ms = int(1000.0 / fps)
        pil_frames[0].save(
            str(out_path),
            save_all=True,
            append_images=pil_frames[1:],
            duration=frame_duration_ms,
            loop=loop,
        )

        return out_path
