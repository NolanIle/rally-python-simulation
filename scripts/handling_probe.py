#!/usr/bin/env python3
"""Measure how a car model behaves, so tuning changes can be compared.

Drives the model headlessly with scripted inputs and prints tables. Run it
before and after a change and diff the output:

    env -u PYTHONPATH uv run python scripts/handling_probe.py > /tmp/before.txt
    ...make changes...
    env -u PYTHONPATH uv run python scripts/handling_probe.py > /tmp/after.txt
    diff /tmp/before.txt /tmp/after.txt

The point of the steering tables is that yaw rate must *increase* with
steering input at every speed. Identical values across a row mean the input
is being ignored.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from rallysim.core.config import CarParams, load_car
from rallysim.core.state import (
    BRAKE_IN,
    HANDBRAKE_IN,
    SPEED,
    STEER_IN,
    THROTTLE_IN,
    X,
    Y,
    YAW,
    make_inputs,
    make_state,
)
from rallysim.physics.kinematic import KinematicBicycle

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAR = REPO_ROOT / "config" / "cars" / "lancia_delta_s4.yaml"
DT = 1.0 / 200.0

SPEEDS_KMH = (30.0, 60.0, 100.0, 140.0)
STEER_FRACTIONS = (0.25, 0.5, 0.75, 1.0)


def drive(model, steer, throttle, brake=0.0, handbrake=0.0, seconds=5.0, dt=DT, state=None):
    """Hold one set of inputs and return the final state (shape (1, state_size))."""
    if state is None:
        state = make_state(1, model.state_size)
    inputs = make_inputs(1)
    inputs[:, STEER_IN] = steer
    inputs[:, THROTTLE_IN] = throttle
    inputs[:, BRAKE_IN] = brake
    inputs[:, HANDBRAKE_IN] = handbrake
    for _ in range(int(round(seconds / dt))):
        state = model.step(state, inputs, dt)
    return state


def _held(steer, throttle, brake):
    """An input array holding one set of controls."""
    inputs = make_inputs(1)
    inputs[:, STEER_IN] = steer
    inputs[:, THROTTLE_IN] = throttle
    inputs[:, BRAKE_IN] = brake
    return inputs


def state_at_speed(model, kmh):
    """A state sitting at the given speed, pointing along +x."""
    state = make_state(1, model.state_size)
    state[0, SPEED] = kmh / 3.6
    return state


def yaw_rate_at(model, kmh, steer):
    """Steady-state yaw rate [deg/s] for a steering input at a fixed speed.

    Runs long enough for the steering actuator to reach its target, then
    measures one step with throttle trimmed to hold speed roughly constant.
    """
    state = state_at_speed(model, kmh)
    inputs = make_inputs(1)
    inputs[:, STEER_IN] = steer
    inputs[:, THROTTLE_IN] = 0.3
    for _ in range(int(round(1.5 / DT))):
        state = model.step(state, inputs, DT)
    after = model.step(state, inputs, DT)
    d_yaw = np.arctan2(np.sin(after[0, YAW] - state[0, YAW]), np.cos(after[0, YAW] - state[0, YAW]))
    return np.degrees(d_yaw / DT), after[0, SPEED]


def section(title):
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def probe_steering_response(model):
    section("1. Yaw rate vs steering input  (MUST increase along each row)")
    header = "  speed  " + "".join(f"{int(f * 100):>8}%" for f in STEER_FRACTIONS) + "   spread"
    print(header)
    print("  " + "-" * (len(header) - 2))
    worst_spread = float("inf")
    for kmh in SPEEDS_KMH:
        rates = [yaw_rate_at(model, kmh, f)[0] for f in STEER_FRACTIONS]
        spread = rates[-1] / rates[0] if rates[0] > 1e-9 else float("nan")
        worst_spread = min(worst_spread, spread)
        cells = "".join(f"{r:9.2f}" for r in rates)
        print(f"  {kmh:5.0f}  {cells}   {spread:6.2f}x")
    print()
    print(f"  deg/s. Worst spread across speeds: {worst_spread:.2f}x")
    if worst_spread < 1.05:
        print("  >> STEERING IS SATURATED: input has no effect at some speed.")
    return worst_spread


def probe_steering_authority(model):
    section("2. Steering authority: what fraction of the requested yaw you get")
    p = model.params
    print("  speed   wheels ask   grip cap     actual   authority   turn radius")
    print("  " + "-" * 66)
    for kmh in (20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 140.0, 160.0, 180.0):
        v = kmh / 3.6
        ask = np.degrees(v / p.wheelbase * np.tan(p.max_steer))
        cap = np.degrees(p.max_lateral_accel / max(v, 0.1))
        actual, _ = yaw_rate_at(model, kmh, 1.0)
        actual = abs(actual)
        radius = v / np.radians(actual) if actual > 1e-9 else float("inf")
        print(
            f"  {kmh:5.0f}  {ask:11.1f}  {cap:9.1f}  {actual:9.1f}"
            f"  {actual / ask * 100:9.0f}%  {radius:10.1f} m"
        )
    print("  deg/s unless marked. Authority falling with speed is correct understeer;")
    print("  it is only a bug when the *spread* in table 1 collapses to 1.00x.")


def probe_grip_limit(model):
    section("3. Lateral acceleration must never exceed the grip limit")
    limit = model.params.max_lateral_accel
    print(f"  configured max_lateral_accel: {limit:.2f} m/s^2  ({limit / 9.81:.2f} g)")
    print()
    print("  speed    peak lateral      margin   used   ok")
    print("  " + "-" * 52)
    ok_all = True
    worst_used = 1.0
    for kmh in SPEEDS_KMH:
        rate, speed = yaw_rate_at(model, kmh, 1.0)
        lateral = abs(np.radians(rate)) * speed
        ok = lateral <= limit + 1e-6
        ok_all = ok_all and ok
        used = lateral / limit
        # Only judge grip usage well into the saturated regime. The smooth
        # saturation approaches the cap asymptotically, so at lower speeds a
        # figure below 100% is the curve itself, not an extra limiter.
        if kmh >= 100.0:
            worst_used = min(worst_used, used)
        print(
            f"  {kmh:5.0f}  {lateral:11.3f}    {limit - lateral:9.3f}  {used:5.0%}   "
            f"{'yes' if ok else 'NO -- VIOLATED'}"
        )
    print()
    print("  'used' is how much of the available grip the car actually reaches at")
    print("  full lock. Well under 100% at speed means something other than grip")
    print("  is limiting the steering, which feels like driving a ship.")
    return ok_all, worst_used


def probe_performance(model):
    section("4. Straight-line performance sanity (should be unchanged by steering work)")
    state = make_state(1, model.state_size)
    inputs = make_inputs(1)
    inputs[:, THROTTLE_IN] = 1.0
    t = 0.0
    t_100 = None
    while t < 60.0:
        state = model.step(state, inputs, DT)
        t += DT
        if t_100 is None and state[0, SPEED] >= 100 / 3.6:
            t_100 = t
    print(f"  0-100 km/h:        {t_100:.2f} s" if t_100 else "  0-100 km/h: never reached")
    print(f"  top speed:         {state[0, SPEED] * 3.6:.1f} km/h")

    state = state_at_speed(model, 100.0)
    start_x = state[0, X]
    t = 0.0
    while state[0, SPEED] > 0.01 and t < 20.0:
        state = drive(model, 0.0, 0.0, brake=1.0, seconds=DT, state=state)
        t += DT
    print(f"  100-0 km/h:        {t:.2f} s over {state[0, X] - start_x:.1f} m")

    coast = drive(model, 0.0, 0.0, seconds=1.0, state=state_at_speed(model, 80.0))
    hand = drive(model, 0.0, 0.0, handbrake=1.0, seconds=1.0, state=state_at_speed(model, 80.0))
    foot = drive(model, 0.0, 0.0, brake=1.0, seconds=1.0, state=state_at_speed(model, 80.0))
    print(
        f"  from 80 km/h, 1 s: coast {coast[0, SPEED] * 3.6:5.1f}  "
        f"handbrake {hand[0, SPEED] * 3.6:5.1f}  footbrake {foot[0, SPEED] * 3.6:5.1f} km/h"
    )


def probe_reverse(model):
    section("5. Reverse")
    state = drive(model, 0.0, 0.0, brake=1.0, seconds=8.0)
    v = state[0, SPEED] * 3.6
    cap = getattr(model.params, "max_reverse_speed", None)
    if v >= -0.01:
        print(f"  braking from rest for 8 s gives {v:.2f} km/h -> no reverse gear")
        return
    print(f"  braking from rest for 8 s gives {v:.2f} km/h")
    if cap is not None:
        print(f"  configured cap: {-cap * 3.6:.2f} km/h   respected: {v >= -cap * 3.6 - 0.01}")

    # Measure one step each way, so the comparison cannot be confused by yaw
    # wrapping past +/-180 deg over a long run.
    fwd = drive(model, 1.0, 0.3, seconds=1.5, state=state_at_speed(model, 20.0))
    fwd_rate = np.degrees(model.step(fwd, _held(1.0, 0.3, 0.0), DT)[0, YAW] - fwd[0, YAW]) / DT
    rev = drive(model, 1.0, 0.0, brake=1.0, seconds=1.5)
    rev_rate = np.degrees(model.step(rev, _held(1.0, 0.0, 1.0), DT)[0, YAW] - rev[0, YAW]) / DT
    print(
        f"  same steering input: forward yaw rate {fwd_rate:+8.2f} deg/s, "
        f"reverse {rev_rate:+8.2f} deg/s"
    )
    print("  (opposite signs = the car pivots the other way when backing up)")


def probe_timestep(model):
    section("6. Timestep convergence (should settle as dt shrinks)")
    print("      dt            x            y        yaw")
    print("  " + "-" * 46)
    for divisor in (30, 60, 120, 240, 1000):
        dt = 1.0 / divisor
        state = state_at_speed(model, 60.0)
        inputs = make_inputs(1)
        inputs[:, STEER_IN] = 1.0
        inputs[:, THROTTLE_IN] = 1.0
        for _ in range(int(round(3.0 / dt))):
            state = model.step(state, inputs, dt)
        print(
            f"  1/{divisor:<5d}  {state[0, X]:11.3f}  {state[0, Y]:11.3f}"
            f"  {np.degrees(state[0, YAW]):9.2f}"
        )


def probe_slip(model):
    section("7. Slip angle (zero means the car is on rails and cannot drift)")
    state = state_at_speed(model, 80.0)
    inputs = make_inputs(1)
    inputs[:, STEER_IN] = 1.0
    inputs[:, THROTTLE_IN] = 1.0
    inputs[:, HANDBRAKE_IN] = 1.0
    peak = 0.0
    for _ in range(int(round(3.0 / DT))):
        before = state.copy()
        state = model.step(state, inputs, DT)
        dx = state[0, X] - before[0, X]
        dy = state[0, Y] - before[0, Y]
        if abs(dx) + abs(dy) > 1e-12:
            course = np.arctan2(dy, dx)
            slip = np.arctan2(np.sin(course - state[0, YAW]), np.cos(course - state[0, YAW]))
            peak = max(peak, abs(np.degrees(slip)))
    print(f"  peak slip angle during hard cornering: {peak:.4f} deg")
    if peak < 1e-3:
        print("  -> kinematic model, as documented. Drift needs the dynamic model.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--car", type=Path, default=DEFAULT_CAR, help="car config file")
    args = parser.parse_args(argv)

    params: CarParams = load_car(args.car)
    model = KinematicBicycle(params)

    print(f"car:  {params.name}")
    print(f"file: {args.car}")
    print(
        f"key params: max_steer={np.degrees(params.max_steer):.0f} deg, "
        f"steer_rate={np.degrees(params.steer_rate):.0f} deg/s, "
        f"max_lateral_accel={params.max_lateral_accel:.1f} m/s^2, "
        f"wheelbase={params.wheelbase:.2f} m"
    )

    spread = probe_steering_response(model)
    probe_steering_authority(model)
    grip_ok, grip_used = probe_grip_limit(model)
    probe_performance(model)
    probe_reverse(model)
    probe_timestep(model)
    probe_slip(model)

    section("Summary")
    print(
        f"  input range usable (want > 1.10x):  {spread:.2f}x  "
        f"{'PASS' if spread > 1.10 else 'FAIL -- steering feels on/off'}"
    )
    print(
        f"  grip actually used (want > 90%):     {grip_used:.0%}   "
        f"{'PASS' if grip_used > 0.9 else 'FAIL -- something is over-limiting the steering'}"
    )
    print(
        f"  grip limit respected:               {'PASS' if grip_ok else 'FAIL -- exceeds the limit'}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
