"""Keyboard input ramping.

The gamepad paths need a real device, so only the keyboard fallback is
covered here.
"""

from pathlib import Path

import pygame
import pytest

from rallysim.core.config import load_car
from rallysim.core.state import BRAKE_IN, STEER_IN, THROTTLE_IN
from rallysim.viz.input import InputHandler

CAR_FILE = Path(__file__).resolve().parents[2] / "config" / "cars" / "lancia_delta_s4.yaml"
DT = 1.0 / 60.0


class _HeldKeys:
    """Stands in for pygame's key state with a fixed set of keys held down."""

    def __init__(self, *held):
        self.held = set(held)

    def __getitem__(self, key):
        return key in self.held


@pytest.fixture
def handler(monkeypatch):
    monkeypatch.setattr(pygame.joystick, "init", lambda: None)
    monkeypatch.setattr(pygame.joystick, "get_count", lambda: 0)
    return InputHandler()


def hold(handler, monkeypatch, *keys, seconds):
    """Hold keys for a while and return the final input array."""
    monkeypatch.setattr(pygame.key, "get_pressed", lambda: _HeldKeys(*keys))
    inputs = handler.read(DT)
    elapsed = DT
    while elapsed < seconds:
        inputs = handler.read(DT)
        elapsed += DT
    return inputs


def test_keyboard_steering_is_not_the_bottleneck(handler, monkeypatch):
    """Keyboard steering must reach full lock about as fast as the car can steer.

    A slower ramp means the input layer, not the vehicle, limits turn-in --
    which made the car feel unresponsive at speed.
    """
    car = load_car(CAR_FILE)
    car_lock_time = car.max_steer / car.steer_rate
    ramp_time = 1.0 / InputHandler.KEY_STEER_RATE
    assert ramp_time <= car_lock_time * 1.5, (
        f"keyboard takes {ramp_time * 1000:.0f} ms to reach full lock but the car "
        f"only needs {car_lock_time * 1000:.0f} ms"
    )


def test_steering_ramps_up_and_centres(handler, monkeypatch):
    partial = hold(handler, monkeypatch, pygame.K_LEFT, seconds=1.0 / 24.0)
    assert 0.0 < partial[0, STEER_IN] < 1.0  # ramped, not instant

    full = hold(handler, monkeypatch, pygame.K_LEFT, seconds=0.5)
    assert full[0, STEER_IN] == pytest.approx(1.0)

    centred = hold(handler, monkeypatch, seconds=0.5)
    assert centred[0, STEER_IN] == pytest.approx(0.0)


def test_left_and_right_are_opposite(handler, monkeypatch):
    left = hold(handler, monkeypatch, pygame.K_LEFT, seconds=0.5)[0, STEER_IN]
    hold(handler, monkeypatch, seconds=0.5)
    right = hold(handler, monkeypatch, pygame.K_RIGHT, seconds=0.5)[0, STEER_IN]
    assert left == pytest.approx(-right)


def test_wasd_matches_the_arrow_keys(handler, monkeypatch):
    a = hold(handler, monkeypatch, pygame.K_a, seconds=0.5)[0, STEER_IN]
    hold(handler, monkeypatch, seconds=0.5)
    left = hold(handler, monkeypatch, pygame.K_LEFT, seconds=0.5)[0, STEER_IN]
    assert a == pytest.approx(left)


def test_pedals_ramp_independently(handler, monkeypatch):
    inputs = hold(handler, monkeypatch, pygame.K_UP, seconds=0.5)
    assert inputs[0, THROTTLE_IN] == pytest.approx(1.0)
    assert inputs[0, BRAKE_IN] == pytest.approx(0.0)

    inputs = hold(handler, monkeypatch, pygame.K_DOWN, seconds=1.0)
    assert inputs[0, BRAKE_IN] == pytest.approx(1.0)
    assert inputs[0, THROTTLE_IN] == pytest.approx(0.0)


def test_no_keys_means_no_input(handler, monkeypatch):
    inputs = hold(handler, monkeypatch, seconds=0.5)
    assert inputs.sum() == pytest.approx(0.0)
