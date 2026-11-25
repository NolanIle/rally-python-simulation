import itertools
from pathlib import Path
import numpy as np
import pytest
from rallysim.core.config import load_car
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

CAR_FILE = Path(__file__).resolve().parents[2] / "config" / "cars" / "lancia_delta_s4.yaml"
DT = 1.0 / 200.0


@pytest.fixture
def model() -> KinematicBicycle:
    return KinematicBicycle(load_car(CAR_FILE))


def run(model, steer, throttle, brake=0.0, handbrake=0.0, seconds=5.0, n_cars=1, state=None):
    if state is None:
        state = make_state(n_cars, model.state_size)
    inputs = make_inputs(n_cars)
    inputs[:, STEER_IN] = steer
    inputs[:, THROTTLE_IN] = throttle
    inputs[:, BRAKE_IN] = brake
    inputs[:, HANDBRAKE_IN] = handbrake
    for _ in range(int(round(seconds / DT))):
        state = model.step(state, inputs, DT)
    return state


def test_straight_acceleration(model):
    s = run(model, steer=0.0, throttle=1.0)
    assert s[0, SPEED] > 10.0
    assert s[0, X] > 0.0
    assert abs(s[0, Y]) < 1e-9
    assert abs(s[0, YAW]) < 1e-9


def test_zero_to_hundred_is_plausible(model):
    state = make_state(1, model.state_size)
    inputs = make_inputs(1)
    inputs[:, THROTTLE_IN] = 1.0
    t = 0.0
    while state[0, SPEED] < 100 / 3.6 and t < 20.0:
        state = model.step(state, inputs, DT)
        t += DT
    assert 2.5 < t < 5.0


def test_braking_stops_the_car(model):
    """Braking kills forward speed. Held at rest the brake becomes reverse."""
    s = run(model, steer=0.0, throttle=1.0, seconds=3.0)
    assert s[0, SPEED] > 0.0
    s = run(model, steer=0.0, throttle=0.0, brake=1.0, seconds=10.0, state=s)
    assert s[0, SPEED] <= 0.0


def test_handbrake_slows_the_car(model):
    moving = run(model, steer=0.0, throttle=1.0, seconds=3.0)
    coasting = run(model, steer=0.0, throttle=0.0, seconds=1.0, state=moving)
    handbraking = run(model, steer=0.0, throttle=0.0, handbrake=1.0, seconds=1.0, state=moving)
    assert abs(handbraking[0, SPEED]) < abs(coasting[0, SPEED])


def test_lateral_acceleration_is_limited(model):
    state = run(model, steer=0.0, throttle=1.0, seconds=4.0)
    inputs = make_inputs(1)
    inputs[:, STEER_IN] = 1.0
    inputs[:, THROTTLE_IN] = 0.3
    limit = model.params.max_lateral_accel
    for _ in range(400):
        new = model.step(state, inputs, DT)
        d_yaw = new[0, YAW] - state[0, YAW]
        d_yaw = np.arctan2(np.sin(d_yaw), np.cos(d_yaw))
        lateral = new[0, SPEED] * abs(d_yaw) / DT
        assert lateral <= limit + 1e-6
        state = new


def at_speed(model, kmh):
    """A state travelling at the given speed along +x."""
    state = make_state(1, model.state_size)
    state[0, SPEED] = kmh / 3.6
    return state


def steady_yaw_rate(model, kmh, steer):
    """Yaw rate [rad/s] once the steering has settled at a roughly fixed speed."""
    state = at_speed(model, kmh)
    inputs = make_inputs(1)
    inputs[:, STEER_IN] = steer
    inputs[:, THROTTLE_IN] = 0.3
    for _ in range(int(round(1.5 / DT))):
        state = model.step(state, inputs, DT)
    after = model.step(state, inputs, DT)
    d_yaw = after[0, YAW] - state[0, YAW]
    return abs(np.arctan2(np.sin(d_yaw), np.cos(d_yaw))) / DT


def test_steering_response_is_monotonic(model):
    """More steering must always give more yaw rate, at every speed.

    Regression test for the hard grip clip, which made every input above the
    limit produce an identical yaw rate and left the steering dead at speed.
    """
    for kmh in (30.0, 60.0, 100.0, 140.0):
        rates = [steady_yaw_rate(model, kmh, f) for f in (0.25, 0.5, 0.75, 1.0)]
        for smaller, larger in itertools.pairwise(rates):
            assert larger > smaller, f"yaw rate did not increase at {kmh:.0f} km/h: {rates}"


def test_steering_input_range_stays_usable(model):
    """Full lock must yaw the car meaningfully harder than a quarter of it.

    Grip saturation compresses the input range as speed rises -- that is the
    physics -- but the spread must not collapse towards 1.0, which is what
    made the steering feel like an on/off switch.
    """
    for kmh in (30.0, 60.0, 100.0, 140.0):
        small = steady_yaw_rate(model, kmh, 0.25)
        full = steady_yaw_rate(model, kmh, 1.0)
        assert full / small > 1.1, f"response too compressed at {kmh:.0f} km/h"


def test_steering_authority_decreases_with_speed(model):
    """The same input yaws the car less when it is going faster: understeer."""
    rates = [steady_yaw_rate(model, kmh, 1.0) for kmh in (30.0, 60.0, 100.0, 140.0)]
    for faster, slower in itertools.pairwise(rates):
        assert slower < faster


def test_cornering_uses_the_available_grip(model):
    """At speed the car should corner near its grip limit, not well under it.

    Guards against stacking another limiter on top of the grip limit: an
    earlier speed-sensitive steering curve held the car to 0.95 g when it was
    configured for 1.1 g, which felt like driving a ship.
    """
    limit = model.params.max_lateral_accel
    for kmh in (80.0, 120.0, 140.0):
        state = at_speed(model, kmh)
        inputs = make_inputs(1)
        inputs[:, STEER_IN] = 1.0
        inputs[:, THROTTLE_IN] = 0.3
        for _ in range(int(round(1.5 / DT))):
            state = model.step(state, inputs, DT)
        after = model.step(state, inputs, DT)
        d_yaw = after[0, YAW] - state[0, YAW]
        rate = abs(np.arctan2(np.sin(d_yaw), np.cos(d_yaw))) / DT
        lateral = rate * after[0, SPEED]
        assert lateral > 0.9 * limit, (
            f"only using {lateral / limit:.0%} of available grip at {kmh:.0f} km/h"
        )


def test_reverse_engages_and_is_capped(model):
    """Holding the brake at rest backs the car up, no faster than the cap."""
    s = run(model, steer=0.0, throttle=0.0, brake=1.0, seconds=2.0)
    assert s[0, SPEED] < 0.0
    s = run(model, steer=0.0, throttle=0.0, brake=1.0, seconds=20.0)
    assert s[0, SPEED] >= -model.params.max_reverse_speed - 1e-9


def test_reverse_steers_the_other_way(model):
    """A reversing car pivots opposite to a forward one on the same input."""
    forward = run(model, steer=1.0, throttle=0.3, seconds=2.0, state=at_speed(model, 20.0))
    reverse = run(model, steer=1.0, throttle=0.0, brake=1.0, seconds=2.0)
    assert np.sign(forward[0, YAW]) == -np.sign(reverse[0, YAW])


def test_throttle_recovers_from_reverse(model):
    """The throttle stops a reversing car and pulls it forward again."""
    s = run(model, steer=0.0, throttle=0.0, brake=1.0, seconds=4.0)
    assert s[0, SPEED] < 0.0
    s = run(model, steer=0.0, throttle=1.0, seconds=3.0, state=s)
    assert s[0, SPEED] > 0.0


def test_stationary_car_does_not_creep(model):
    """With no input at all the car stays exactly where it is."""
    s = run(model, steer=0.0, throttle=0.0, seconds=10.0)
    assert s[0, SPEED] == 0.0
    assert s[0, X] == 0.0


def test_left_right_symmetry(model):
    left = run(model, steer=0.5, throttle=0.5)
    right = run(model, steer=-0.5, throttle=0.5)
    np.testing.assert_allclose(left[0, X], right[0, X], atol=1e-9)
    np.testing.assert_allclose(left[0, Y], -right[0, Y], atol=1e-9)
    np.testing.assert_allclose(np.sin(left[0, YAW]), -np.sin(right[0, YAW]), atol=1e-9)
    np.testing.assert_allclose(np.cos(left[0, YAW]), np.cos(right[0, YAW]), atol=1e-9)


def test_deterministic(model):
    a = run(model, steer=0.3, throttle=0.7)
    b = run(model, steer=0.3, throttle=0.7)
    np.testing.assert_array_equal(a, b)


def test_batch_matches_single_cars(model):
    steers = np.array([0.3, -0.7])
    batch = run(model, steer=steers, throttle=0.8, n_cars=2)
    for i, steer in enumerate(steers):
        single = run(model, steer=float(steer), throttle=0.8)
        np.testing.assert_allclose(batch[i], single[0])
