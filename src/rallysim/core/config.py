"""Load YAML configuration into validated, immutable parameter objects.

Config files may use convenient units (degrees, kW); everything is
converted to SI here, so the rest of the code only ever sees SI units.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from rallysim.physics.tyre import TyreParams

MODELS = ("kinematic", "dynamic")


class ConfigError(ValueError):
    """Raised when a configuration file is missing values or has invalid ones."""


@dataclass(frozen=True)
class CarParams:
    name: str
    mass: float  # [kg]
    max_power: float  # [W]
    wheelbase: float  # [m]
    max_steer: float  # [rad]
    steer_rate: float  # [rad/s]
    max_accel: float  # traction limit [m/s^2]
    max_brake: float  # [m/s^2]
    handbrake_decel: float  # [m/s^2], placeholder until tire slip exists
    max_lateral_accel: float  # KINEMATIC MODEL ONLY: grip limit in corners [m/s^2]
    drag_coeff: float  # [1/m], drag deceleration = drag_coeff * v^2
    rolling_resistance: float  # [m/s^2]
    max_reverse_speed: float  # [m/s], reverse is capped here
    reverse_accel_fraction: float  # 0..1, reverse acceleration as a share of max_accel
    # The rest is used by the dynamic model only; the kinematic model has no
    # tyres and gets its cornering limit from max_lateral_accel above.
    yaw_inertia: float  # [kg m^2], resistance to rotating about the vertical
    cg_to_front: float  # [m], centre of gravity to front axle
    cg_to_rear: float  # [m], centre of gravity to rear axle
    tyre_front: TyreParams
    tyre_rear: TyreParams


_CAR_KEYS = (
    "name",
    "mass_kg",
    "max_power_kw",
    "wheelbase_m",
    "max_steer_deg",
    "steer_rate_deg_s",
    "max_accel_m_s2",
    "max_brake_m_s2",
    "handbrake_decel_m_s2",
    "max_lateral_accel_m_s2",
    "drag_coeff_per_m",
    "rolling_resistance_m_s2",
    "max_reverse_speed_m_s",
    "reverse_accel_fraction",
    "yaw_inertia_kg_m2",
    "cg_to_front_m",
    "cg_to_rear_m",
    "tyres",
)

_TYRE_KEYS = ("stiffness", "shape", "friction", "curvature")


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping at the top level")
    return data


def _require_keys(data: dict[str, Any], keys: tuple[str, ...], path: Path) -> None:
    missing = [k for k in keys if k not in data]
    if missing:
        raise ConfigError(f"{path}: missing keys {missing}")


def _load_tyre(data: dict[str, Any], axle: str, path: Path) -> TyreParams:
    """Read one axle's tyre coefficients out of the `tyres` block."""
    tyres = data["tyres"]
    if not isinstance(tyres, dict) or axle not in tyres:
        raise ConfigError(f"{path}: tyres must be a mapping with 'front' and 'rear'")
    values = tyres[axle]
    if not isinstance(values, dict):
        raise ConfigError(f"{path}: tyres.{axle} must be a mapping")
    missing = [k for k in _TYRE_KEYS if k not in values]
    if missing:
        raise ConfigError(f"{path}: tyres.{axle} missing keys {missing}")
    tyre = TyreParams(
        stiffness=float(values["stiffness"]),
        shape=float(values["shape"]),
        friction=float(values["friction"]),
        curvature=float(values["curvature"]),
    )
    for name, value in (
        ("stiffness", tyre.stiffness),
        ("shape", tyre.shape),
        ("friction", tyre.friction),
    ):
        if not value > 0:
            raise ConfigError(f"{path}: tyres.{axle}.{name} must be positive")
    # Past 1.0 the Magic Formula stops being monotonic before its peak and the
    # force curve folds back on itself, which is not a tyre any more.
    if not 0.0 <= tyre.curvature <= 1.0:
        raise ConfigError(f"{path}: tyres.{axle}.curvature must be between 0 and 1")
    return tyre


def load_car(path: str | Path) -> CarParams:
    """Load a car file and return its parameters in SI units."""
    path = Path(path)
    data = _read_yaml(path)
    _require_keys(data, _CAR_KEYS, path)

    params = CarParams(
        name=str(data["name"]),
        mass=float(data["mass_kg"]),
        max_power=float(data["max_power_kw"]) * 1000.0,
        wheelbase=float(data["wheelbase_m"]),
        max_steer=math.radians(float(data["max_steer_deg"])),
        steer_rate=math.radians(float(data["steer_rate_deg_s"])),
        max_accel=float(data["max_accel_m_s2"]),
        max_brake=float(data["max_brake_m_s2"]),
        handbrake_decel=float(data["handbrake_decel_m_s2"]),
        max_lateral_accel=float(data["max_lateral_accel_m_s2"]),
        drag_coeff=float(data["drag_coeff_per_m"]),
        rolling_resistance=float(data["rolling_resistance_m_s2"]),
        max_reverse_speed=float(data["max_reverse_speed_m_s"]),
        reverse_accel_fraction=float(data["reverse_accel_fraction"]),
        yaw_inertia=float(data["yaw_inertia_kg_m2"]),
        cg_to_front=float(data["cg_to_front_m"]),
        cg_to_rear=float(data["cg_to_rear_m"]),
        tyre_front=_load_tyre(data, "front", path),
        tyre_rear=_load_tyre(data, "rear", path),
    )

    positive = {
        "mass_kg": params.mass,
        "max_power_kw": params.max_power,
        "wheelbase_m": params.wheelbase,
        "max_steer_deg": params.max_steer,
        "steer_rate_deg_s": params.steer_rate,
        "max_accel_m_s2": params.max_accel,
        "max_brake_m_s2": params.max_brake,
        "max_lateral_accel_m_s2": params.max_lateral_accel,
        "max_reverse_speed_m_s": params.max_reverse_speed,
        "reverse_accel_fraction": params.reverse_accel_fraction,
        "yaw_inertia_kg_m2": params.yaw_inertia,
        "cg_to_front_m": params.cg_to_front,
        "cg_to_rear_m": params.cg_to_rear,
    }
    bad = [k for k, v in positive.items() if not v > 0]
    if bad:
        raise ConfigError(f"{path}: values must be positive: {bad}")
    if params.reverse_accel_fraction > 1.0:
        raise ConfigError(f"{path}: reverse_accel_fraction must be <= 1")
    # The two axle distances are the wheelbase split about the centre of
    # gravity, so they have to add up or the car is geometrically impossible.
    axle_sum = params.cg_to_front + params.cg_to_rear
    if not math.isclose(axle_sum, params.wheelbase, rel_tol=1e-6):
        raise ConfigError(
            f"{path}: cg_to_front_m + cg_to_rear_m ({axle_sum:.3f}) must equal "
            f"wheelbase_m ({params.wheelbase:.3f})"
        )
    non_negative = {
        "handbrake_decel_m_s2": params.handbrake_decel,
        "drag_coeff_per_m": params.drag_coeff,
        "rolling_resistance_m_s2": params.rolling_resistance,
    }
    bad = [k for k, v in non_negative.items() if v < 0]
    if bad:
        raise ConfigError(f"{path}: values must not be negative: {bad}")

    return params


@dataclass(frozen=True)
class SimParams:
    """How to simulate, as opposed to what car to simulate."""

    model: str
    physics_dt: float  # [s], fixed physics step
    # The dynamic model divides by speed to get slip angles, so it is unusable
    # at walking pace and hands over to no-slip behaviour below this band.
    blend_speed_low: float  # [m/s], fully kinematic below this
    blend_speed_high: float  # [m/s], fully dynamic above this


_SIM_KEYS = ("model", "physics_dt_s", "blend_speed_low_m_s", "blend_speed_high_m_s")


def load_sim(path: str | Path) -> SimParams:
    """Load a simulation settings file."""
    path = Path(path)
    data = _read_yaml(path)
    _require_keys(data, _SIM_KEYS, path)

    model = str(data["model"])
    if model not in MODELS:
        raise ConfigError(f"{path}: model must be one of {list(MODELS)}, got {model!r}")

    params = SimParams(
        model=model,
        physics_dt=float(data["physics_dt_s"]),
        blend_speed_low=float(data["blend_speed_low_m_s"]),
        blend_speed_high=float(data["blend_speed_high_m_s"]),
    )

    if not params.physics_dt > 0:
        raise ConfigError(f"{path}: physics_dt_s must be positive")
    if params.blend_speed_low < 0:
        raise ConfigError(f"{path}: blend_speed_low_m_s must not be negative")
    if params.blend_speed_high <= params.blend_speed_low:
        raise ConfigError(
            f"{path}: blend_speed_high_m_s ({params.blend_speed_high}) must be above "
            f"blend_speed_low_m_s ({params.blend_speed_low})"
        )

    return params
