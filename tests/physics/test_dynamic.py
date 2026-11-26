from pathlib import Path

import numpy as np
import pytest
from rallysim.core.config import load_car
from rallysim.core.state import (
    BRAKE_IN,
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
    body_slip,
    make_inputs,
    make_state,
)
from rallysim.physics.dynamic import DynamicBicycle
from rallysim.physics.kinematic import KinematicBicycle

CAR_FILE = Path(__file__).resolve().parents[2] / "config" / "cars" / "lancia_delta_s4.yaml"
DT = 1.0 / 200.0


@pytest.fixture
def model() -> DynamicBicycle:
    return DynamicBicycle(load_car(CAR_FILE))


def at_speed(model, kmh, n_cars=1):
    state = make_state(n_cars, model.state_size)
    state[:, SPEED] = kmh / 3.6
    return state


def run(
    model,
    steer=0.0,
    throttle=0.0,
    brake=0.0,
    handbrake=0.0,
    seconds=2.0,
    state=None,
    n_cars=1,
):
    """Hold one set of inputs; returns the final state and the peaks seen."""
    if state is None:
        state = make_state(n_cars, model.state_size)
    inputs = make_inputs(n_cars)
    inputs[:, STEER_IN] = steer
    inputs[:, THROTTLE_IN] = throttle
    inputs[:, BRAKE_IN] = brake
    inputs[:, HANDBRAKE_IN] = handbrake
    peak_slip = 0.0
    peak_yaw = 0.0
    for _ in range(int(round(seconds / DT))):
        state = model.step(state, inputs, DT)
        peak_slip = max(peak_slip, abs(body_slip(state[0])))
        peak_yaw = max(peak_yaw, abs(state[0, YAW_RATE]))
    return state, peak_slip, peak_yaw


def test_the_car_can_slide(model):
    """The whole reason this model exists: the kinematic one has zero slip."""
    _, peak_slip, _ = run(model, steer=1.0, throttle=0.3, state=at_speed(model, 100.0))
    assert peak_slip > np.radians(0.5)

    kinematic = KinematicBicycle(load_car(CAR_FILE))
    kin_state = make_state(1, kinematic.state_size)
    kin_state[0, SPEED] = 100 / 3.6
    inputs = make_inputs(1)
    inputs[:, STEER_IN] = 1.0
    inputs[:, THROTTLE_IN] = 0.3
    for _ in range(400):
        kin_state = kinematic.step(kin_state, inputs, DT)
    assert body_slip(kin_state[0]) == 0.0


def test_handbrake_breaks_the_kinematic_yaw_ceiling(model):
    """Regression guard for the sluggishness that motivated this model.

    The kinematic model's yaw rate is pinned to path curvature, so at 100 km/h
    and about 1.1 g it cannot exceed roughly 20 deg/s however it is driven.
    Taking grip away from the rear axle must beat that comfortably.
    """
    _, _, peak_yaw = run(
        model, steer=0.6, throttle=0.3, handbrake=1.0, seconds=1.0, state=at_speed(model, 100.0)
    )
    assert peak_yaw > np.radians(30.0)


def test_handbrake_rotates_more_than_plain_cornering(model):
    """The handbrake must rotate the car, not just slow it down."""
    _, _, cornering_yaw = run(
        model, steer=0.6, throttle=0.3, seconds=1.0, state=at_speed(model, 100.0)
    )
    _, _, handbrake_yaw = run(
        model, steer=0.6, throttle=0.3, handbrake=1.0, seconds=1.0, state=at_speed(model, 100.0)
    )
    assert handbrake_yaw > 1.5 * cornering_yaw


def test_normal_cornering_is_stable(model):
    """Steady steering must settle, not diverge into a spin."""
    state, peak_slip, _ = run(
        model, steer=0.5, throttle=0.35, seconds=5.0, state=at_speed(model, 100.0)
    )
    assert abs(body_slip(state[0])) < np.radians(15.0)
    assert peak_slip < np.radians(20.0)
    assert np.isfinite(state).all()


def test_a_slide_recovers_when_the_handbrake_is_released(model):
    """Slip must decay once the rear axle has its grip back."""
    sliding, _, _ = run(
        model, steer=0.6, throttle=0.3, handbrake=1.0, seconds=0.4, state=at_speed(model, 100.0)
    )
    assert abs(body_slip(sliding[0])) > np.radians(3.0)
    settled, _, _ = run(model, steer=0.0, throttle=0.3, seconds=4.0, state=sliding)
    assert abs(body_slip(settled[0])) < abs(body_slip(sliding[0]))


def test_a_stopped_car_does_not_rotate_or_slide(model):
    """A car that slides to a halt must not keep spinning on the spot.

    Saturated tyres provide no damping of their own, so it is the low-speed
    blend that ends a spin: it scales lateral velocity away and hands the yaw
    rate over to the kinematic value, which is zero at a standstill.
    """
    sliding, _, _ = run(
        model, steer=1.0, throttle=0.5, handbrake=1.0, seconds=1.0, state=at_speed(model, 100.0)
    )
    stopped, _, _ = run(model, brake=0.0, throttle=0.0, seconds=30.0, state=sliding)
    assert abs(stopped[0, SPEED]) < 0.5
    assert abs(stopped[0, YAW_RATE]) < np.radians(1.0)
    assert abs(stopped[0, VY]) < 0.05


def test_idle_car_stays_put(model):
    state, _, _ = run(model, seconds=5.0)
    assert state[0, SPEED] == 0.0
    assert state[0, X] == 0.0
    assert state[0, Y] == 0.0
    assert state[0, YAW_RATE] == pytest.approx(0.0)


def test_low_speed_blend_is_continuous(model):
    """Driving down through the handover band must not kick.

    The car coasts from 40 km/h to a stop on a steady steering input, which is
    how a driver meets the band. Comparing separate runs started at different
    speeds would instead measure turn-in transients, which differ for reasons
    that have nothing to do with the blend.
    """
    state = at_speed(model, 40.0)
    inputs = make_inputs(1)
    inputs[:, STEER_IN] = 0.5
    previous = None
    worst = 0.0
    for _ in range(4000):
        state = model.step(state, inputs, DT)
        if state[0, SPEED] < 0.5:
            break
        rate = state[0, YAW_RATE]
        if previous is not None:
            worst = max(worst, abs(rate - previous))
        previous = rate
    assert np.isfinite(state).all()
    # Per 5 ms step. A visible kick would be an order of magnitude larger.
    assert worst < np.radians(5.0), f"yaw rate jumps {np.degrees(worst):.2f} deg/s in one step"


def test_low_speed_has_no_phantom_slip(model):
    """At walking pace slip angles are fiction, so the car must not slide."""
    _, peak_slip, _ = run(model, steer=1.0, throttle=0.1, seconds=2.0, state=at_speed(model, 5.0))
    assert peak_slip < np.radians(5.0)


def test_sustained_cornering_respects_the_tyre_limit(model):
    """Once settled, the car cannot corner harder than its tyres allow.

    Only the steady state is checked. Turning in is a transient where both
    axles briefly pull the same way before any body slip has developed, which
    legitimately peaks above the steady-state limit for a fraction of a second.
    """
    p = model.params
    # The whole car can only pull what both axles together can make, which is
    # not simply the friction coefficient: the weight sits towards the rear, so
    # the two axles carry different loads.
    weight = p.mass * 9.81
    load_front = weight * p.cg_to_rear / p.wheelbase
    load_rear = weight * p.cg_to_front / p.wheelbase
    limit = (p.tyre_front.friction * load_front + p.tyre_rear.friction * load_rear) / p.mass

    state = at_speed(model, 120.0)
    inputs = make_inputs(1)
    inputs[:, STEER_IN] = 1.0
    inputs[:, THROTTLE_IN] = 0.3
    settled = []
    for step in range(2000):
        state = model.step(state, inputs, DT)
        if step > 1200:  # past the turn-in transient
            settled.append(abs(state[0, YAW_RATE] * state[0, SPEED]))
    worst = max(settled)
    # yaw_rate * speed is only the centripetal part, so it reads a little high
    # while the car is sliding; allow a small margin over the force bound.
    assert worst <= limit * 1.15, f"sustained lateral {worst:.2f} exceeds tyre limit {limit:.2f}"


def test_cornering_does_not_ring(model):
    """A steady corner must settle, not oscillate."""
    state = at_speed(model, 120.0)
    inputs = make_inputs(1)
    inputs[:, STEER_IN] = 1.0
    inputs[:, THROTTLE_IN] = 0.3
    settled = []
    for step in range(1600):
        state = model.step(state, inputs, DT)
        if step > 1200:
            settled.append(abs(state[0, YAW_RATE] * state[0, SPEED]))
    swing = max(settled) - min(settled)
    assert swing < 5.0, f"lateral acceleration swings by {swing:.2f} m/s^2 in a steady corner"


def steady_yaw_rate(model, kmh, steer_input, seconds=6.0):
    """Yaw rate once a steady steering input has settled [rad/s].

    Throttle and brake are driven to hold the target speed. A fixed throttle
    would not do: a harder-cornering car sheds more speed, so the comparison
    would be between different speeds rather than between steering inputs.
    """
    target = kmh / 3.6
    state = at_speed(model, kmh)
    inputs = make_inputs(1)
    inputs[:, STEER_IN] = steer_input
    for _ in range(int(round(seconds / DT))):
        error = target - state[0, SPEED]
        inputs[:, THROTTLE_IN] = np.clip(error * 0.5, 0.0, 1.0)
        inputs[:, BRAKE_IN] = np.clip(-error * 0.5, 0.0, 1.0)
        state = model.step(state, inputs, DT)
    return abs(state[0, YAW_RATE])


def test_more_steering_turns_the_car_harder(model):
    """Full input must be the *best* way to turn, not the worst.

    Found by driving the sandbox: a tyre makes peak force at a few degrees of
    slip and less beyond it, so mapping full input onto full lock dragged the
    front tyre far past its peak. At 126 km/h that gave half the yaw rate of a
    fifth of the input -- the car pushed straight on and left the road.
    """
    for kmh in (60.0, 100.0, 140.0):
        gentle = steady_yaw_rate(model, kmh, 0.25)
        full = steady_yaw_rate(model, kmh, 1.0)
        assert full > gentle, (
            f"at {kmh:.0f} km/h full lock yaws {np.degrees(full):.1f} deg/s but a "
            f"quarter input yaws {np.degrees(gentle):.1f} deg/s -- steering is "
            f"past the tyre's grip peak"
        )


def test_full_lock_is_available_when_slow(model):
    """Capping the steering must not cost low-speed manoeuvrability."""
    state, _, _ = run(model, steer=1.0, throttle=0.2, seconds=1.0, state=at_speed(model, 10.0))
    assert state[0, STEER] == pytest.approx(model.params.max_steer, rel=1e-3)


def test_reverse_engages_and_is_capped(model):
    state, _, _ = run(model, brake=1.0, seconds=2.0)
    assert state[0, SPEED] < 0.0
    state, _, _ = run(model, brake=1.0, seconds=20.0)
    assert state[0, SPEED] >= -model.params.max_reverse_speed - 1e-9


def test_reverse_steers_the_other_way(model):
    forward, _, _ = run(model, steer=1.0, throttle=0.3, seconds=2.0, state=at_speed(model, 20.0))
    reverse, _, _ = run(model, steer=1.0, brake=1.0, seconds=2.0)
    assert np.sign(forward[0, YAW]) == -np.sign(reverse[0, YAW])


def test_throttle_recovers_from_reverse(model):
    state, _, _ = run(model, brake=1.0, seconds=4.0)
    assert state[0, SPEED] < 0.0
    state, _, _ = run(model, throttle=1.0, seconds=3.0, state=state)
    assert state[0, SPEED] > 0.0


def test_straight_line_performance_is_plausible(model):
    """The engine model is carried over, so these should match the old figures."""
    state = make_state(1, model.state_size)
    inputs = make_inputs(1)
    inputs[:, THROTTLE_IN] = 1.0
    t = 0.0
    while state[0, SPEED] < 100 / 3.6 and t < 20.0:
        state = model.step(state, inputs, DT)
        t += DT
    assert 2.5 < t < 5.0
    assert abs(state[0, Y]) < 1e-9  # dead straight
    assert body_slip(state[0]) == pytest.approx(0.0)


def test_deterministic(model):
    a, _, _ = run(model, steer=0.3, throttle=0.7, state=at_speed(model, 60.0))
    b, _, _ = run(model, steer=0.3, throttle=0.7, state=at_speed(model, 60.0))
    np.testing.assert_array_equal(a, b)


def test_left_right_symmetry(model):
    left, _, _ = run(model, steer=0.5, throttle=0.5, state=at_speed(model, 80.0))
    right, _, _ = run(model, steer=-0.5, throttle=0.5, state=at_speed(model, 80.0))
    np.testing.assert_allclose(left[0, X], right[0, X], atol=1e-9)
    np.testing.assert_allclose(left[0, Y], -right[0, Y], atol=1e-9)
    np.testing.assert_allclose(left[0, VY], -right[0, VY], atol=1e-9)
    np.testing.assert_allclose(left[0, YAW_RATE], -right[0, YAW_RATE], atol=1e-9)


def test_batch_matches_single_cars(model):
    """The model must stay vectorised: no Python branching per car."""
    steers = np.array([0.3, -0.7])
    batch = at_speed(model, 80.0, n_cars=2)
    inputs = make_inputs(2)
    inputs[:, STEER_IN] = steers
    inputs[:, THROTTLE_IN] = 0.4
    for _ in range(400):
        batch = model.step(batch, inputs, DT)

    for i, steer in enumerate(steers):
        single, _, _ = run(
            model, steer=float(steer), throttle=0.4, seconds=2.0, state=at_speed(model, 80.0)
        )
        np.testing.assert_allclose(batch[i], single[0], atol=1e-9)


def test_timestep_convergence(model):
    """Explicit Euler must be stable and converge as the step shrinks."""
    results = []
    for divisor in (60, 200, 1000):
        dt = 1.0 / divisor
        state = at_speed(model, 100.0)
        inputs = make_inputs(1)
        inputs[:, STEER_IN] = 0.8
        inputs[:, THROTTLE_IN] = 0.3
        for _ in range(int(round(2.0 / dt))):
            state = model.step(state, inputs, dt)
        results.append(state[0, YAW_RATE])
        assert np.isfinite(state).all()
    # Halving the step again should change the answer less than the first jump.
    assert abs(results[2] - results[1]) < abs(results[1] - results[0])
