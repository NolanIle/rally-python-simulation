"""Tyre forces: the Pacejka "Magic Formula" and a friction circle.

These are plain functions over numpy arrays with no knowledge of any vehicle
layout, so a two-axle model and a future four-wheel model can both use them --
the four-wheel version just passes four loads instead of two.

The Magic Formula gives a lateral force that rises with slip angle, *peaks*,
then falls away again. That peak is what makes sliding possible: past it, more
slip means less grip, so the rear can break away and be caught again. A model
without the falloff (a linear tyre, or the hard grip clip the kinematic model
uses) can never drift.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TyreParams:
    """Magic Formula coefficients for one axle.

    stiffness (B): how quickly force builds with slip. Higher = sharper, more
        responsive; gravel is lower than tarmac.
    shape (C): where the peak sits and how pointy it is.
    friction (D): peak force as a fraction of the vertical load, i.e. the
        friction coefficient. 1.1 means the tyre can pull 1.1 g at full load.
    curvature (E): how much force is lost past the peak. This is the drift
        knob: E around 0.5 loses about a third of the peak force by 30 deg of
        slip (slidy but catchable), while E near 1.0 barely falls off at all
        and makes a slide impossible to hold.
    """

    stiffness: float
    shape: float
    friction: float
    curvature: float


def lateral_force(slip_angle: np.ndarray, load: np.ndarray, tyre: TyreParams) -> np.ndarray:
    """Lateral force [N] resisting a slip angle, for a given vertical load [N].

    Returns a force opposing the slip, so the sign is opposite to the slip
    angle. Peak force scales with load, which is what lets load transfer
    change the balance of the car once it is modelled.
    """
    peak = tyre.friction * load
    scaled = tyre.stiffness * slip_angle
    shaped = scaled - tyre.curvature * (scaled - np.arctan(scaled))
    return -peak * np.sin(tyre.shape * np.arctan(shaped))


def peak_slip_angle(tyre: TyreParams) -> float:
    """Slip angle [rad] at which this tyre makes its most lateral force.

    Steering a wheel much past this is counterproductive: the force curve is
    falling by then, so more lock gives less grip. Found numerically because
    the Magic Formula has no tidy closed-form peak.
    """
    candidates = np.radians(np.linspace(0.01, 45.0, 2000))
    forces = np.abs(lateral_force(candidates, np.ones_like(candidates), tyre))
    return float(candidates[np.argmax(forces)])


def friction_circle(
    force_long: np.ndarray, force_lat: np.ndarray, limit: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Scale a combined force back onto the friction circle.

    A tyre has one grip budget shared between accelerating, braking and
    cornering. Asking for both at once gives less of each, which is why
    braking into a corner washes the nose wide and why throttle steers the car.

    Forces already inside the circle are returned unchanged.
    """
    magnitude = np.hypot(force_long, force_lat)
    safe_limit = np.maximum(limit, 1e-9)
    # Only shrink; never scale a small force up to the limit.
    scale = np.where(magnitude > safe_limit, safe_limit / np.maximum(magnitude, 1e-9), 1.0)
    return force_long * scale, force_lat * scale
