"""Grip-limited kinematic bicycle model: no tire slip.

A placeholder that moves believably enough to build the sandbox, input
and rendering around it. Cornering is capped by a lateral grip limit
(the car turns less than steered when going too fast: understeer), and
acceleration is limited by traction at low speed and power at high
speed. It is replaced by the drift model later, and stays available as
the simple fallback model.
"""

from __future__ import annotations
import numpy as np
from rallysim.core.config import CarParams
from rallysim.core.state import (
    BRAKE_IN,
    COMMON_STATE_SIZE,
    HANDBRAKE_IN,
    SPEED,
    STEER,
    STEER_IN,
    THROTTLE_IN,
    X,
    Y,
    YAW,
)


class KinematicBicycle:
    state_size = COMMON_STATE_SIZE

    def __init__(self, params: CarParams) -> None:
        self.params = params

    def step(self, state: np.ndarray, inputs: np.ndarray, dt: float) -> np.ndarray:
        p = self.params

        steer_cmd = np.clip(inputs[:, STEER_IN], -1.0, 1.0)
        throttle = np.clip(inputs[:, THROTTLE_IN], 0.0, 1.0)
        brake = np.clip(inputs[:, BRAKE_IN], 0.0, 1.0)
        handbrake = np.clip(inputs[:, HANDBRAKE_IN], 0.0, 1.0)

        # Steering moves towards the commanded angle at a limited rate.
        target = steer_cmd * p.max_steer
        max_change = p.steer_rate * dt
        steer = state[:, STEER] + np.clip(target - state[:, STEER], -max_change, max_change)

        # Engine: traction-limited at low speed, power-limited at high speed.
        v = state[:, SPEED]
        engine_accel = np.minimum(p.max_accel, p.max_power / (p.mass * np.maximum(v, 1.0)))
        accel = throttle * engine_accel - p.drag_coeff * v * v - p.rolling_resistance
        accel -= brake * p.max_brake + handbrake * p.handbrake_decel
        v_new = np.maximum(v + accel * dt, 0.0)

        # Cornering: the yaw rate the wheels ask for, capped by lateral grip.
        yaw_rate = v_new / p.wheelbase * np.tan(steer)
        max_yaw_rate = p.max_lateral_accel / np.maximum(v_new, 0.1)
        yaw_rate = np.clip(yaw_rate, -max_yaw_rate, max_yaw_rate)

        yaw = state[:, YAW] + yaw_rate * dt
        yaw = (yaw + np.pi) % (2.0 * np.pi) - np.pi

        new_state = np.empty_like(state)
        new_state[:, X] = state[:, X] + v_new * np.cos(yaw) * dt
        new_state[:, Y] = state[:, Y] + v_new * np.sin(yaw) * dt
        new_state[:, YAW] = yaw
        new_state[:, SPEED] = v_new
        new_state[:, STEER] = steer
        return new_state
