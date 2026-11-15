"""Top-down camera: converts world coordinates [m] to screen pixels."""

from __future__ import annotations
import math
import numpy as np


class Camera:
    MIN_SCALE = 2.0
    MAX_SCALE = 40.0

    def __init__(
            self,
            screen_size: tuple[int, int],
            pixels_per_metre: float = 8.0,
            smoothing: float = 5.0,
    ) -> None:
        self.width, self.height = screen_size
        self.scale = pixels_per_metre
        self.smoothing = smoothing
        self.cx = 0.0
        self.cy = 0.0

    def snap(self, x: float, y: float) -> None:
        self.cx, self.cy = x, y

    def follow(self, x: float, y: float, dt: float) -> None:
        """Move smoothly towards (x, y), independent of frame rate."""
        alpha = 1.0 - math.exp(-self.smoothing * dt)
        self.cx += (x - self.cx) * alpha
        self.cy += (y - self.cy) * alpha

    def zoom(self, factor: float) -> None:
        self.scale = min(max(self.scale * factor, self.MIN_SCALE), self.MAX_SCALE)

    def to_screen(self, x: float, y: float) -> tuple[float, float]:
        # Screen y grows downwards, world y grows upwards.
        return (
            self.width / 2 + (x - self.cx) * self.scale,
            self.height / 2 - (y - self.cy) * self.scale,
        )

    def to_screen_many(self, points: np.ndarray) -> list[tuple[float, float]]:
        sx = self.width / 2 + (points[:, 0] - self.cx) * self.scale
        sy = self.height / 2 - (points[:, 1] - self.cy) * self.scale
        return list(zip(sx.tolist(), sy.tolist()))

    def view_half_extent(self) -> tuple[float, float]:
        """Half the visible width and height in metres."""
        return self.width / 2 / self.scale, self.height / 2 / self.scale
