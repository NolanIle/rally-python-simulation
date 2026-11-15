"""A hand-made test road for the sandbox, until the stage generator exists."""

from __future__ import annotations
import math
import numpy as np

# (length [m], curvature [1/m]); positive curvature turns left
SEGMENTS: list[tuple[float, float]] = [
    (150.0, 0.0),
    (60.0, 1.0 / 40.0),
    (80.0, 0.0),
    (90.0, -1.0 / 25.0),
    (40.0, 0.0),
    (120.0, 1.0 / 60.0),
    (100.0, 0.0),
    (50.0, -1.0 / 15.0),
    (150.0, 0.0),
]


def make_test_road(spacing: float = 1.0) -> np.ndarray:
    """Return the road centreline as an (N, 2) array of points [m]."""
    x = y = heading = 0.0
    points = [(x, y)]
    for length, curvature in SEGMENTS:
        n = max(1, int(round(length / spacing)))
        ds = length / n
        for _ in range(n):
            heading += curvature * ds
            x += math.cos(heading) * ds
            y += math.sin(heading) * ds
            points.append((x, y))
    return np.array(points, dtype=np.float64)


def road_edges(centreline: np.ndarray, width: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (left_edge, right_edge) arrays offset from the centreline."""
    tangent = np.gradient(centreline, axis=0)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-9)
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])  # points left
    half = width / 2.0
    return centreline + normal * half, centreline - normal * half
