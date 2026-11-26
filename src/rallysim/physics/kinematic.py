"""Grip-limited kinematic bicycle model: no tire slip.

The simple fallback model, kept alongside the dynamic one and selected
with `model: kinematic` in a sim config. Cornering is limited by lateral
grip (the car turns less than steered when going too fast: understeer),
and acceleration is limited by traction at low speed and power at high
speed.

The grip limit saturates the yaw rate smoothly rather than clipping it.
A hard clip would make every steering input beyond the limit produce the
same yaw rate, leaving the steering dead above about 20 km/h.

Grip is the only thing limiting cornering here: at speed the steering
geometry asks for far more yaw than any tyre can deliver (full lock at
120 km/h would need 27 g), so the radius comes straight from
`max_lateral_accel` -- about 103 m at 120 km/h for 1.1 g.

The car has no slip angle: it always travels exactly where it points, so
it cannot drift. That needs `dynamic.py`.
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

        # Full lock is available at any speed; the grip limit below decides how
        # much of it the car can use. The dynamic model caps the angle instead.
        target = steer_cmd * p.max_steer
        max_change = p.steer_rate * dt
        steer = state[:, STEER] + np.clip(target - state[:, STEER], -max_change, max_change)

        # Reverse: the brake doubles as reverse gear once the car has stopped,
        # so holding it backs out of an overshot corner. "Stopped" needs a small
        # tolerance, otherwise the car flaps between gears either side of zero.
        stopped = speed < REVERSE_ENGAGE_SPEED
        in_reverse = (v < 0.0) | (stopped & (brake > 0.0) & (throttle <= 0.0))

        direction = np.where(in_reverse, -1.0, 1.0)

        # Engine: traction-limited at low speed, power-limited at high speed.
        # Reverse pulls weaker and is the brake pedal's job, not the throttle's.
        engine_accel = np.minimum(p.max_accel, p.max_power / (p.mass * np.maximum(speed, 1.0)))
        drive = np.where(
            in_reverse,
            brake * p.max_accel * p.reverse_accel_fraction,
            throttle * engine_accel,
        )

        # Resistance opposes the direction the car is actually moving, which is
        # not always the direction the drivetrain is pushing.
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

        # Coasting must stop at zero rather than reversing through it.
        crossed_zero = (v * v_new) < 0.0
        coasting_to_rest = crossed_zero & (drive <= 1e-12)
        v_new = np.where(coasting_to_rest, 0.0, v_new)
        v_new = np.where(in_reverse, np.maximum(v_new, -p.max_reverse_speed), v_new)
        v_new = np.where(~in_reverse, np.maximum(v_new, 0.0), v_new)

        # The wheels ask for a yaw rate and grip limits what the car can take.
        # Saturating smoothly keeps the response monotonic in steering input
        # while staying strictly under the limit; see the module docstring for
        # why a hard clip does not.
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
