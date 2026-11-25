"""Grip-limited kinematic bicycle model: no tire slip.

A placeholder that moves believably enough to build the sandbox, input
and rendering around it. Cornering is limited by lateral grip (the car
turns less than steered when going too fast: understeer), and
acceleration is limited by traction at low speed and power at high
speed. It is replaced by the drift model later, and stays available as
the simple fallback model.

One detail matters for how it feels to drive: the grip limit saturates
the yaw rate smoothly instead of clipping it. A hard clip makes every
steering input beyond the limit produce the same yaw rate, which leaves
the steering dead above about 20 km/h.

Lateral grip is the only thing limiting how hard the car can turn. At
speed the steering geometry asks for far more yaw than any tire can
deliver (full lock at 120 km/h would need 27 g), so cornering is always
grip-bound and the resulting radius comes straight from
`max_lateral_accel`: about 103 m at 120 km/h for 1.1 g. That is correct
physics, not sluggishness -- if the car feels too slow to turn, raise
`max_lateral_accel_m_s2`, because nothing else is holding it back.

The car has no slip angle: it always travels exactly where it points.
Drifting needs the dynamic model.
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


REVERSE_ENGAGE_SPEED = 0.2  # [m/s] below this the brake engages reverse


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

        v = state[:, SPEED]
        speed = np.abs(v)

        # Steering moves towards the commanded angle at a limited rate. Full
        # lock is available at any speed; the grip limit below decides how much
        # of it the car can actually use.
        target = steer_cmd * p.max_steer
        max_change = p.steer_rate * dt
        steer = state[:, STEER] + np.clip(target - state[:, STEER], -max_change, max_change)

        # Reverse: the brake doubles as reverse gear once the car has stopped,
        # so holding it backs out of an overshot corner. "Stopped" needs a small
        # tolerance, otherwise the car flaps between gears either side of zero.
        stopped = speed < REVERSE_ENGAGE_SPEED
        in_reverse = (v < 0.0) | (stopped & (brake > 0.0) & (throttle <= 0.0))

        # Direction of travel the drivetrain is working in: +1 forward, -1 back.
        direction = np.where(in_reverse, -1.0, 1.0)

        # Engine: traction-limited at low speed, power-limited at high speed.
        # Reverse pulls weaker and is the brake pedal's job, not the throttle's.
        engine_accel = np.minimum(p.max_accel, p.max_power / (p.mass * np.maximum(speed, 1.0)))
        drive = np.where(
            in_reverse,
            brake * p.max_accel * p.reverse_accel_fraction,
            throttle * engine_accel,
        )

        # Resistance always opposes travel, so it acts along -direction.
        resistance = p.rolling_resistance + p.drag_coeff * speed * speed
        # Whichever pedal is not driving acts as the brake: the throttle stops a
        # reversing car, the brake stops a forward one.
        resistance += np.where(in_reverse, throttle * p.max_brake, brake * p.max_brake)
        resistance += handbrake * p.handbrake_decel

        accel = direction * drive - np.sign(v) * resistance
        # At a standstill np.sign(v) is 0, which would let resistance vanish and
        # the car creep off on rolling resistance alone. Oppose the drive instead.
        accel = np.where(stopped & (speed < 1e-9), direction * drive, accel)
        v_new = v + accel * dt

        # Resistance must not push the car past zero into the other direction,
        # and reverse is capped.
        crossed_zero = (v * v_new) < 0.0
        coasting_to_rest = crossed_zero & (drive <= 1e-12)
        v_new = np.where(coasting_to_rest, 0.0, v_new)
        v_new = np.where(in_reverse, np.maximum(v_new, -p.max_reverse_speed), v_new)
        v_new = np.where(~in_reverse, np.maximum(v_new, 0.0), v_new)

        # Cornering. The wheels ask for a yaw rate; lateral grip limits how much
        # of it the car can actually take. Saturate smoothly rather than clipping:
        # a hard clip makes every steering input above the limit produce the same
        # yaw rate, so the steering goes dead. This keeps the response monotonic
        # in steering input while staying strictly under the grip limit.
        speed_new = np.abs(v_new)
        asked = v_new / p.wheelbase * np.tan(steer)
        max_yaw_rate = p.max_lateral_accel / np.maximum(speed_new, 0.1)
        yaw_rate = asked * max_yaw_rate / (np.abs(asked) + max_yaw_rate)

        yaw = state[:, YAW] + yaw_rate * dt
        yaw = (yaw + np.pi) % (2.0 * np.pi) - np.pi

        new_state = np.empty_like(state)
        new_state[:, X] = state[:, X] + v_new * np.cos(yaw) * dt
        new_state[:, Y] = state[:, Y] + v_new * np.sin(yaw) * dt
        new_state[:, YAW] = yaw
        new_state[:, SPEED] = v_new
        new_state[:, STEER] = steer
        return new_state
