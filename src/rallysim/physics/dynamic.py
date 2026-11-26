"""Dynamic bicycle model with tyre slip: the car can slide.

Each axle is lumped into one tyre at the centre of the car's width, so this is
still a "bicycle", but unlike the kinematic model the tyres now generate forces
from slip angles instead of steering the car geometrically. That difference is
the whole point:

* The car has a body slip angle, so it can travel in a direction other than the
  one it points. That is a slide.
* Yaw rate is no longer tied to path curvature. Drop the rear grip (handbrake)
  and the car rotates far faster than it corners, which is what rally driving
  looks like. The kinematic model is capped near 20 deg/s at 100 km/h; this one
  reaches 60-150 deg/s with the rear loose.
* Grip is one budget shared between cornering and accelerating (the friction
  circle), so throttle and brakes steer the car.

Slip angle is ``atan(vy / vx)``, so the model is meaningless near zero speed:
at 0.1 m/s it reports 83 deg of slip and saturated tyres. It therefore blends
to no-slip behaviour below a few m/s, which also ends spins.

Forces are integrated with explicit Euler. Results converge between dt = 1/60
and 1/1000, including through a 70 deg slide, so the fixed 200 Hz step has
plenty of margin.
"""

from __future__ import annotations

import numpy as np

from rallysim.core.config import CarParams
from rallysim.core.state import (
    BRAKE_IN,
    DYNAMIC_STATE_SIZE,
    HANDBRAKE_IN,
    SPEED,
    STEER,
    STEER_IN,
    THROTTLE_IN,
    VY,
    X,
    Y,
    YAW,
    YAW_RATE,
)
from rallysim.physics.tyre import friction_circle, lateral_force, peak_slip_angle

GRAVITY = 9.81

REVERSE_ENGAGE_SPEED = 0.2  # [m/s] below this the brake engages reverse

# Fraction of rear lateral grip lost when the handbrake is fully applied. This
# is what makes the tail step out, so it is the main handbrake tuning knob.
HANDBRAKE_GRIP_LOSS = 0.65


class DynamicBicycle:
    state_size = DYNAMIC_STATE_SIZE

    def __init__(
        self,
        params: CarParams,
        blend_speed_low: float = 3.0,
        blend_speed_high: float = 6.0,
    ) -> None:
        self.params = params
        self.blend_speed_low = blend_speed_low
        self.blend_speed_high = blend_speed_high
        # Cached because finding the peak is a numerical search.
        self._peak_slip = peak_slip_angle(params.tyre_front)

    def step(self, state: np.ndarray, inputs: np.ndarray, dt: float) -> np.ndarray:
        p = self.params

        steer_cmd = np.clip(inputs[:, STEER_IN], -1.0, 1.0)
        throttle = np.clip(inputs[:, THROTTLE_IN], 0.0, 1.0)
        brake = np.clip(inputs[:, BRAKE_IN], 0.0, 1.0)
        handbrake = np.clip(inputs[:, HANDBRAKE_IN], 0.0, 1.0)

        vx = state[:, SPEED]
        vy = state[:, VY]
        yaw_rate = state[:, YAW_RATE]
        speed = np.abs(vx)

        # Full travel maps onto the angle the front tyre can use, not onto full
        # lock: past its peak slip a tyre makes *less* force, so mapping full
        # input onto 30 deg of lock at speed would turn the car more slowly
        # than a fifth of that input. A steady corner at the grip limit needs
        # the Ackermann angle plus part of the peak slip -- part, because body
        # slip already supplies the rest. Below ~30 km/h this exceeds max_steer
        # and full lock comes back for hairpins.
        grip_limit = max(p.tyre_front.friction, p.tyre_rear.friction) * GRAVITY
        usable_steer = (
            p.wheelbase * grip_limit / np.maximum(speed, 1.0) ** 2 + 0.5 * self._peak_slip
        )
        max_angle = np.minimum(usable_steer, p.max_steer)
        target = steer_cmd * max_angle
        max_change = p.steer_rate * dt
        steer = state[:, STEER] + np.clip(target - state[:, STEER], -max_change, max_change)

        # The brake doubles as reverse gear once the car has stopped, matching
        # the kinematic model so the two feel the same at a standstill.
        stopped = speed < REVERSE_ENGAGE_SPEED
        in_reverse = (vx < 0.0) | (stopped & (brake > 0.0) & (throttle <= 0.0))
        direction = np.where(in_reverse, -1.0, 1.0)

        engine_accel = np.minimum(p.max_accel, p.max_power / (p.mass * np.maximum(speed, 1.0)))
        drive = np.where(
            in_reverse,
            brake * p.max_accel * p.reverse_accel_fraction,
            throttle * engine_accel,
        )

        resistance = p.rolling_resistance + p.drag_coeff * speed * speed
        # Whichever pedal is not driving acts as the brake.
        resistance += np.where(in_reverse, throttle * p.max_brake, brake * p.max_brake)
        resistance += handbrake * p.handbrake_decel

        accel_long = direction * drive - np.sign(vx) * resistance
        # At a standstill np.sign is 0, which would let the car creep off on
        # rolling resistance alone; oppose the drive instead.
        accel_long = np.where(stopped & (speed < 1e-9), direction * drive, accel_long)

        # Axle loads: the static split from the centre of gravity, plus
        # longitudinal transfer so braking loads the front and power loads the
        # rear. Per-axle for now; a four-wheel model would split these sideways
        # as well.
        weight = p.mass * GRAVITY
        static_front = weight * p.cg_to_rear / p.wheelbase
        static_rear = weight * p.cg_to_front / p.wheelbase
        cg_height = 0.45  # [m] estimate; only the order of magnitude matters here
        transfer = p.mass * accel_long * cg_height / p.wheelbase
        load_front = np.maximum(static_front - transfer, 0.0)
        load_rear = np.maximum(static_rear + transfer, 0.0)

        # Floor the divisor: the slip angles are meaningless near zero speed,
        # and the blend further down discards them there anyway.
        vx_safe = np.maximum(speed, self.blend_speed_low)
        slip_front = np.arctan2(vy + yaw_rate * p.cg_to_front, vx_safe) - steer
        slip_rear = np.arctan2(vy - yaw_rate * p.cg_to_rear, vx_safe)

        # Taking grip from the rear axle is what makes the handbrake rotate the
        # car, rather than merely slowing it.
        rear_grip = 1.0 - HANDBRAKE_GRIP_LOSS * handbrake
        force_front = lateral_force(slip_front, load_front, p.tyre_front)
        force_rear = lateral_force(slip_rear, load_rear * rear_grip, p.tyre_rear)

        # Friction circle: grip spent driving or braking is unavailable for
        # cornering, so hard braking into a corner washes the nose wide.
        long_per_axle = p.mass * accel_long / 2.0
        _, force_front = friction_circle(
            long_per_axle, force_front, p.tyre_front.friction * load_front
        )
        _, force_rear = friction_circle(
            long_per_axle, force_rear, p.tyre_rear.friction * load_rear * rear_grip
        )

        accel_lat = (force_front * np.cos(steer) + force_rear) / p.mass - yaw_rate * vx
        yaw_accel = (
            p.cg_to_front * force_front * np.cos(steer) - p.cg_to_rear * force_rear
        ) / p.yaw_inertia

        vx_new = vx + (accel_long + yaw_rate * vy) * dt
        vy_new = vy + accel_lat * dt
        yaw_rate_new = yaw_rate + yaw_accel * dt

        # Brakes must not drag the car through zero into the other direction.
        crossed_zero = (vx * vx_new) < 0.0
        coasting_to_rest = crossed_zero & (drive <= 1e-12)
        vx_new = np.where(coasting_to_rest, 0.0, vx_new)
        vx_new = np.where(in_reverse, np.maximum(vx_new, -p.max_reverse_speed), vx_new)
        vx_new = np.where(~in_reverse, np.maximum(vx_new, 0.0), vx_new)

        # Fade out the slip at low speed, where the angles above are fiction,
        # and fall back to the kinematic result: the car goes where it points.
        # This is also what ends a spin. Saturated tyres give no damping of
        # their own, so without the blend a car that slid to a halt would keep
        # rotating; instead vy is scaled away and the yaw rate becomes the
        # kinematic one, which is zero at a standstill.
        speed_new = np.abs(vx_new)
        blend = np.clip(
            (speed_new - self.blend_speed_low) / (self.blend_speed_high - self.blend_speed_low),
            0.0,
            1.0,
        )
        kinematic_yaw_rate = vx_new / p.wheelbase * np.tan(steer)
        vy_new = blend * vy_new
        yaw_rate_new = blend * yaw_rate_new + (1.0 - blend) * kinematic_yaw_rate

        yaw = state[:, YAW] + yaw_rate_new * dt
        yaw = (yaw + np.pi) % (2.0 * np.pi) - np.pi
        # Velocity is in the body frame, so rotate it into the world.
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)

        new_state = np.empty_like(state)
        new_state[:, X] = state[:, X] + (vx_new * cos_yaw - vy_new * sin_yaw) * dt
        new_state[:, Y] = state[:, Y] + (vx_new * sin_yaw + vy_new * cos_yaw) * dt
        new_state[:, YAW] = yaw
        new_state[:, SPEED] = vx_new
        new_state[:, STEER] = steer
        new_state[:, VY] = vy_new
        new_state[:, YAW_RATE] = yaw_rate_new
        return new_state
