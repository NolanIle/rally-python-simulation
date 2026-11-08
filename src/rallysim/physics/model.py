"""Interface every vehicle model implements."""

from __future__ import annotations
from typing import Protocol
import numpy as np


class VehicleModel(Protocol):
    """A vehicle model advances the state of many cars at once.

    state:  (n_cars, state_size) array, common fields first (see core.state)
    inputs: (n_cars, INPUT_SIZE) array of driver inputs
    Returns a new state array; the input arrays are not modified.
    """

    state_size: int

    def step(self, state: np.ndarray, inputs: np.ndarray, dt: float) -> np.ndarray: ...
