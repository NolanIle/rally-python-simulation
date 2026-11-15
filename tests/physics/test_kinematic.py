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
    s = run(model, steer=0.0, throttle=1.0, seconds=3.0)
    s = run(model, steer=0.0, throttle=0.0, brake=1.0, seconds=10.0, state=s)
    assert s[0, SPEED] == 0.0


def test_handbrake_slows_the_car(model):
    moving = run(model, steer=0.0, throttle=1.0, seconds=3.0)
    coasting = run(model, steer=0.0, throttle=0.0, seconds=1.0, state=moving)
    handbraking = run(model, steer=0.0, throttle=0.0, handbrake=1.0, seconds=1.0, state=moving)
    assert handbraking[0, SPEED] < coasting[0, SPEED]


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
