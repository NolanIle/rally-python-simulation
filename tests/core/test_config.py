from pathlib import Path

import pytest
import yaml
from rallysim.core.config import ConfigError, load_car, load_sim

CONFIG = Path(__file__).resolve().parents[2] / "config"
CAR_FILE = CONFIG / "cars" / "lancia_delta_s4.yaml"
SIM_FILE = CONFIG / "sim" / "default.yaml"
KINEMATIC_SIM = CONFIG / "sim" / "kinematic.yaml"


def write(tmp_path, data, name="car.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def car_data():
    return yaml.safe_load(CAR_FILE.read_text(encoding="utf-8"))


def sim_data():
    return yaml.safe_load(SIM_FILE.read_text(encoding="utf-8"))


def test_shipped_configs_load():
    car = load_car(CAR_FILE)
    assert car.name == "Lancia Delta S4"
    assert load_sim(SIM_FILE).model == "dynamic"
    assert load_sim(KINEMATIC_SIM).model == "kinematic"


def test_units_are_converted_to_si():
    car = load_car(CAR_FILE)
    raw = car_data()
    assert car.max_power == raw["max_power_kw"] * 1000.0
    assert car.max_steer == pytest.approx(0.5236, abs=1e-3)  # 30 deg in radians


def test_tyre_params_round_trip():
    car = load_car(CAR_FILE)
    raw = car_data()["tyres"]
    assert car.tyre_front.stiffness == raw["front"]["stiffness"]
    assert car.tyre_front.friction == raw["front"]["friction"]
    assert car.tyre_rear.curvature == raw["rear"]["curvature"]


def test_missing_file_is_reported(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_car(tmp_path / "nope.yaml")


def test_missing_keys_are_listed(tmp_path):
    data = car_data()
    del data["mass_kg"]
    with pytest.raises(ConfigError, match="mass_kg"):
        load_car(write(tmp_path, data))


def test_negative_mass_is_rejected(tmp_path):
    data = car_data()
    data["mass_kg"] = -1
    with pytest.raises(ConfigError, match="positive"):
        load_car(write(tmp_path, data))


def test_axle_distances_must_match_the_wheelbase(tmp_path):
    """A car whose axle split disagrees with its wheelbase is impossible."""
    data = car_data()
    data["cg_to_front_m"] = 1.0
    data["cg_to_rear_m"] = 1.0  # sums to 2.0, wheelbase is 2.44
    with pytest.raises(ConfigError, match="wheelbase"):
        load_car(write(tmp_path, data))


def test_tyre_block_must_have_both_axles(tmp_path):
    data = car_data()
    del data["tyres"]["rear"]
    with pytest.raises(ConfigError, match="front.*rear|rear"):
        load_car(write(tmp_path, data))


def test_tyre_missing_coefficient_is_reported(tmp_path):
    data = car_data()
    del data["tyres"]["front"]["curvature"]
    with pytest.raises(ConfigError, match="curvature"):
        load_car(write(tmp_path, data))


def test_tyre_curvature_is_bounded(tmp_path):
    """Past 1.0 the Magic Formula folds back and stops being a tyre."""
    data = car_data()
    data["tyres"]["front"]["curvature"] = 1.5
    with pytest.raises(ConfigError, match="curvature"):
        load_car(write(tmp_path, data))


def test_unknown_model_is_rejected(tmp_path):
    data = sim_data()
    data["model"] = "magic"
    with pytest.raises(ConfigError, match="kinematic.*dynamic|model"):
        load_sim(write(tmp_path, data, "sim.yaml"))


def test_blend_band_must_be_ordered(tmp_path):
    data = sim_data()
    data["blend_speed_low_m_s"] = 9.0
    data["blend_speed_high_m_s"] = 4.0
    with pytest.raises(ConfigError, match="blend_speed_high"):
        load_sim(write(tmp_path, data, "sim.yaml"))


def test_physics_dt_must_be_positive(tmp_path):
    data = sim_data()
    data["physics_dt_s"] = 0.0
    with pytest.raises(ConfigError, match="physics_dt"):
        load_sim(write(tmp_path, data, "sim.yaml"))


def test_top_level_must_be_a_mapping(tmp_path):
    path = tmp_path / "list.yaml"
    path.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_car(path)
