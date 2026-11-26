import numpy as np
import pytest
from rallysim.physics.tyre import TyreParams, friction_circle, lateral_force

LOAD = np.array([5000.0])
TYRE = TyreParams(stiffness=10.0, shape=1.9, friction=1.1, curvature=0.5)


def force_at(degrees, tyre=TYRE, load=LOAD):
    return float(lateral_force(np.radians(np.array([degrees])), load, tyre)[0])


def test_no_slip_means_no_force():
    assert force_at(0.0) == pytest.approx(0.0)


def test_force_opposes_the_slip():
    """Force resists the slip, so the signs are opposite."""
    assert force_at(5.0) < 0.0
    assert force_at(-5.0) > 0.0


def test_force_is_odd_symmetric():
    for degrees in (1.0, 5.0, 15.0, 40.0):
        assert force_at(degrees) == pytest.approx(-force_at(-degrees))


def test_force_rises_then_falls():
    """The peak is what makes sliding possible: past it, more slip = less grip.

    A curve that only ever rises (a linear tyre) can never let the rear break
    away and be caught again.
    """
    angles = np.linspace(0.1, 45.0, 400)
    forces = np.abs([force_at(a) for a in angles])
    peak = int(np.argmax(forces))
    assert peak > 0, "force peaks at zero slip"
    assert peak < len(angles) - 1, "force never falls off: no slide possible"
    assert forces[-1] < forces[peak]


def test_curvature_controls_the_falloff():
    """`curvature` is the drift knob: low values lose more grip past the peak."""
    falloffs = {}
    for curvature in (0.3, 0.5, 0.9):
        tyre = TyreParams(stiffness=10.0, shape=1.9, friction=1.1, curvature=curvature)
        angles = np.linspace(0.1, 45.0, 400)
        forces = np.abs([force_at(a, tyre=tyre) for a in angles])
        peak = forces.max()
        falloffs[curvature] = (peak - abs(force_at(30.0, tyre=tyre))) / peak
    assert falloffs[0.3] > falloffs[0.5] > falloffs[0.9]


def test_peak_force_scales_with_load():
    """Load transfer can only change the balance if peak force follows load."""
    light = abs(force_at(10.0, load=np.array([2500.0])))
    heavy = abs(force_at(10.0, load=np.array([5000.0])))
    assert heavy == pytest.approx(2.0 * light)


def test_peak_force_respects_the_friction_coefficient():
    """friction is a multiple of load, so it reads as g of grip."""
    angles = np.linspace(0.1, 45.0, 400)
    peak = max(abs(force_at(a)) for a in angles)
    assert peak == pytest.approx(TYRE.friction * LOAD[0], rel=1e-3)


def test_friction_circle_leaves_small_forces_alone():
    fx, fy = friction_circle(np.array([100.0]), np.array([200.0]), np.array([5000.0]))
    assert fx[0] == pytest.approx(100.0)
    assert fy[0] == pytest.approx(200.0)


def test_friction_circle_caps_the_combined_force():
    limit = 5000.0
    fx, fy = friction_circle(np.array([4000.0]), np.array([4000.0]), np.array([limit]))
    assert np.hypot(fx[0], fy[0]) == pytest.approx(limit)


def test_friction_circle_keeps_the_direction():
    fx, fy = friction_circle(np.array([6000.0]), np.array([3000.0]), np.array([5000.0]))
    assert fy[0] / fx[0] == pytest.approx(3000.0 / 6000.0)


def test_friction_circle_handles_zero_limit():
    """A fully unloaded tyre must not produce NaN."""
    fx, fy = friction_circle(np.array([1000.0]), np.array([1000.0]), np.array([0.0]))
    assert np.isfinite([fx[0], fy[0]]).all()


def test_tyre_functions_are_vectorised():
    slips = np.radians(np.array([2.0, -8.0, 25.0]))
    loads = np.array([4000.0, 5000.0, 6000.0])
    forces = lateral_force(slips, loads, TYRE)
    assert forces.shape == (3,)
    for i in range(3):
        single = lateral_force(slips[i : i + 1], loads[i : i + 1], TYRE)
        assert forces[i] == pytest.approx(single[0])
