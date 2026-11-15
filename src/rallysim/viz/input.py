"""Driver input from a gamepad, with keyboard fallback."""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pygame
from rallysim.core.state import BRAKE_IN, HANDBRAKE_IN, STEER_IN, THROTTLE_IN, make_inputs


@dataclass(frozen=True)
class GamepadMapping:
    """Axis and button numbers differ between controllers; check with `rally gamepad-test`."""

    steer_axis: int = 0
    throttle_axis: int = 5
    brake_axis: int = 2
    handbrake_button: int = 0
    deadzone: float = 0.05
    steer_exponent: float = 1.5  # >1 gives finer control near the centre


class InputHandler:
    KEY_STEER_RATE = 3.0  # how fast keyboard steering ramps [1/s]
    KEY_PEDAL_RATE = 4.0  # how fast keyboard pedals ramp [1/s]

    def __init__(self, mapping: GamepadMapping | None = None) -> None:
        self.mapping = mapping or GamepadMapping()
        self.joystick: pygame.joystick.JoystickType | None = None
        self._armed: dict[int, bool] = {}
        self._keys = np.zeros(4)
        pygame.joystick.init()
        if pygame.joystick.get_count() > 0:
            self._connect(0)

    @property
    def source(self) -> str:
        return self.joystick.get_name() if self.joystick is not None else "keyboard"

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.JOYDEVICEADDED and self.joystick is None:
            self._connect(event.device_index)
        elif event.type == pygame.JOYDEVICEREMOVED and self.joystick is not None:
            if event.instance_id == self.joystick.get_instance_id():
                self.joystick = None

    def read(self, dt: float) -> np.ndarray:
        if self.joystick is not None:
            values = self._read_gamepad()
        else:
            values = self._read_keyboard(dt)
        inputs = make_inputs(1)
        inputs[0, STEER_IN], inputs[0, THROTTLE_IN], inputs[0, BRAKE_IN], inputs[0, HANDBRAKE_IN] = values
        return inputs

    def _connect(self, device_index: int) -> None:
        self.joystick = pygame.joystick.Joystick(device_index)
        self._armed = {self.mapping.throttle_axis: False, self.mapping.brake_axis: False}

    def _axis(self, axis: int) -> float:
        assert self.joystick is not None
        if axis >= self.joystick.get_numaxes():
            return 0.0
        return float(self.joystick.get_axis(axis))

    def _button(self, button: int) -> float:
        assert self.joystick is not None
        if button >= self.joystick.get_numbuttons():
            return 0.0
        return float(self.joystick.get_button(button))

    def _trigger(self, axis: int) -> float:
        # Triggers rest at -1 and go to +1, but many report 0 until first touched.
        # Ignore a trigger until it has been seen at rest, so it doesn't start at half.
        raw = self._axis(axis)
        if raw < -0.5:
            self._armed[axis] = True
        if not self._armed.get(axis, False):
            return 0.0
        return float(np.clip((raw + 1.0) / 2.0, 0.0, 1.0))

    def _read_gamepad(self) -> tuple[float, float, float, float]:
        m = self.mapping
        raw = self._axis(m.steer_axis)
        if abs(raw) < m.deadzone:
            raw = 0.0
        # Response curve, then flip sign: stick left (negative) = steer left (positive).
        steer = -float(np.sign(raw) * abs(raw) ** m.steer_exponent)
        return (
            steer,
            self._trigger(m.throttle_axis),
            self._trigger(m.brake_axis),
            self._button(m.handbrake_button),
        )

    def _read_keyboard(self, dt: float) -> tuple[float, float, float, float]:
        keys = pygame.key.get_pressed()
        steer = float(keys[pygame.K_LEFT] or keys[pygame.K_a]) - float(
            keys[pygame.K_RIGHT] or keys[pygame.K_d]
        )
        throttle = float(keys[pygame.K_UP] or keys[pygame.K_w])
        brake = float(keys[pygame.K_DOWN] or keys[pygame.K_s])
        handbrake = float(keys[pygame.K_SPACE])
        targets = np.array([steer, throttle, brake, handbrake])
        rates = (
                np.array([self.KEY_STEER_RATE, self.KEY_PEDAL_RATE, self.KEY_PEDAL_RATE, 20.0]) * dt
        )
        self._keys += np.clip(targets - self._keys, -rates, rates)
        return (
            float(self._keys[0]),
            float(self._keys[1]),
            float(self._keys[2]),
            float(self._keys[3]),
        )
