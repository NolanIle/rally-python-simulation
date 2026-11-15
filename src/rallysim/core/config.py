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
    max_lateral_accel: float  # grip limit in corners [m/s^2]
    drag_coeff: float  # [1/m], drag deceleration = drag_coeff * v^2
    rolling_resistance: float  # [m/s^2]


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
)


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
    }
    bad = [k for k, v in positive.items() if not v > 0]
    if bad:
        raise ConfigError(f"{path}: values must be positive: {bad}")
    non_negative = {
        "handbrake_decel_m_s2": params.handbrake_decel,
        "drag_coeff_per_m": params.drag_coeff,
        "rolling_resistance_m_s2": params.rolling_resistance,
    }
    bad = [k for k, v in non_negative.items() if v < 0]
    if bad:
        raise ConfigError(f"{path}: values must not be negative: {bad}")

    return params