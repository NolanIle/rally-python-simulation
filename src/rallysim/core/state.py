"""Shared layout of vehicle state and driver input arrays.

All quantities are SI: metres, seconds, radians, metres per second.
State arrays have shape (n_cars, model.state_size); every model stores
the common fields below first, then its own extra fields.
Input arrays have shape (n_cars, INPUT_SIZE).
"""

from __future__ import annotations
import numpy as np

# Common state fields
X = 0  # position x [m]
Y = 1  # position y [m]
YAW = 2  # heading [rad], 0 = +x axis, counter-clockwise positive
SPEED = 3  # longitudinal speed [m/s]
STEER = 4  # front wheel steering angle [rad]

COMMON_STATE_SIZE = 5

# Extra fields for models that let the car slide. SPEED above is the
# longitudinal (body-frame) speed in those models, so everything that only
# reads the common fields keeps working.
VY = 5  # lateral velocity, body frame [m/s], positive to the left
YAW_RATE = 6  # [rad/s], counter-clockwise positive

DYNAMIC_STATE_SIZE = 7

# Driver inputs
STEER_IN = 0  # steering command, -1 (full right) .. 1 (full left)
THROTTLE_IN = 1  # 0 .. 1
BRAKE_IN = 2  # 0 .. 1
HANDBRAKE_IN = 3  # 0 .. 1

INPUT_SIZE = 4


def make_state(n_cars: int, state_size: int = COMMON_STATE_SIZE) -> np.ndarray:
    """Return a zeroed state array for n_cars cars."""
    return np.zeros((n_cars, state_size), dtype=np.float64)


def make_inputs(n_cars: int) -> np.ndarray:
    """Return a zeroed input array for n_cars cars."""
    return np.zeros((n_cars, INPUT_SIZE), dtype=np.float64)


def body_slip(row: np.ndarray) -> float:
    """Angle between where one car points and where it is actually going [rad].

    Zero means the car travels exactly along its heading. Models without a VY
    field cannot slide, so they always report zero.
    """
    if row.shape[-1] <= VY:
        return 0.0
    return float(np.arctan2(row[VY], abs(row[SPEED])) if abs(row[SPEED]) > 1e-6 else 0.0)
