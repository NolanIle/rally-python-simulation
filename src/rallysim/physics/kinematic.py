"""Kinematic bicycle model: no tire slip.

A placeholder that moves believably enough to build the sandbox, input
and rendering around it. It is replaced by the drift model later.
"""

from __future__ import annotations

import numpy as np

from rallysim.core.config import CarParams
from rallysim.core.state import (
    BRAKE_IN,
    COMMON_STATE_SIZE,
    SPEED,
    STEER,
    STEER_IN,
    THROTTLE_IN,
    X,
    Y,
    YAW,
)

# TODO: Kinematic model will have to be replaced with dynamic model
class KinematicBicycle:
    state_size = COMMON_STATE_SIZE

    def __init__(self, params: CarParams) -> None:
        self.params = params

    def step(self, state: np.ndarray, inputs: np.ndarray, dt: float) -> np.ndarray:
        p = self.params

        steer_cmd = np.clip(inputs[:, STEER_IN], -1.0, 1.0)
        throttle = np.clip(inputs[:, THROTTLE_IN], 0.0, 1.0)
        brake = np.clip(inputs[:, BRAKE_IN], 0.0, 1.0)

        # Steering moves towards the commanded angle at a limited rate.
        target = steer_cmd * p.max_steer
        max_change = p.steer_rate * dt
        steer = state[:, STEER] + np.clip(target - state[:, STEER], -max_change, max_change)

        # Longitudinal speed: engine, drag, rolling resistance, brakes. No reversing.
        v = state[:, SPEED]
        accel = throttle * p.max_accel - p.drag_coeff * v * v - p.rolling_resistance
        accel -= brake * p.max_brake
        v_new = np.maximum(v + accel * dt, 0.0)

        # Semi-implicit update: heading and position use the new speed and steering.
        yaw_rate = v_new / p.wheelbase * np.tan(steer)
        yaw = state[:, YAW] + yaw_rate * dt
        yaw = (yaw + np.pi) % (2.0 * np.pi) - np.pi

        new_state = np.empty_like(state)
        new_state[:, X] = state[:, X] + v_new * np.cos(yaw) * dt
        new_state[:, Y] = state[:, Y] + v_new * np.sin(yaw) * dt
        new_state[:, YAW] = yaw
        new_state[:, SPEED] = v_new
        new_state[:, STEER] = steer
        return new_state
