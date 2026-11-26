"""Build a vehicle model by name, so callers need not import every model."""

from __future__ import annotations

from rallysim.core.config import MODELS, CarParams, ConfigError, SimParams
from rallysim.physics.dynamic import DynamicBicycle
from rallysim.physics.kinematic import KinematicBicycle
from rallysim.physics.model import VehicleModel


def build_model(sim: SimParams, car: CarParams) -> VehicleModel:
    """Return the model named by the simulation settings."""
    if sim.model == "kinematic":
        return KinematicBicycle(car)
    if sim.model == "dynamic":
        return DynamicBicycle(
            car,
            blend_speed_low=sim.blend_speed_low,
            blend_speed_high=sim.blend_speed_high,
        )
    # load_sim validates this, so reaching here means MODELS grew a name
    # without a matching branch.
    raise ConfigError(f"no model registered for {sim.model!r} (known: {list(MODELS)})")
