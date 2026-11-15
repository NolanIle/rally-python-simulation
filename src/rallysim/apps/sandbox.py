"""Driving sandbox: drive the car yourself on a hand-made test road.

Controls: gamepad (left stick, triggers) or arrows / WASD.
R resets, mouse wheel zooms, Esc quits.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pygame

from rallysim.core.config import load_car
from rallysim.core.state import BRAKE_IN, SPEED, STEER, THROTTLE_IN, X, Y, YAW, HANDBRAKE_IN, make_state
from rallysim.physics.kinematic import KinematicBicycle
from rallysim.physics.model import VehicleModel
from rallysim.viz.camera import Camera
from rallysim.viz.input import InputHandler
from rallysim.viz.renderer import Renderer
from rallysim.viz.telemetry import TelemetryPanel
from rallysim.world.test_road import make_test_road, road_edges

# src/rallysim/apps/sandbox.py -> repository root is three levels up
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CAR = ROOT / "config" / "cars" / "lancia_delta_s4.yaml"

PHYSICS_DT = 1.0 / 200.0  # fixed physics step [s]
MAX_STEPS_PER_FRAME = 20  # avoid a spiral of death if a frame is very slow
SCREEN_SIZE = (1280, 800)
ROAD_WIDTH = 7.0  # [m]


def initial_state(model: VehicleModel, centreline: np.ndarray) -> np.ndarray:
    state = make_state(1, model.state_size)
    state[0, X], state[0, Y] = centreline[0]
    direction = centreline[1] - centreline[0]
    state[0, YAW] = math.atan2(direction[1], direction[0])
    return state


def run(car: Path | None = None) -> None:
    params = load_car(car or DEFAULT_CAR)
    model = KinematicBicycle(params)
    centreline = make_test_road()
    left, right = road_edges(centreline, ROAD_WIDTH)

    pygame.init()
    screen = pygame.display.set_mode(SCREEN_SIZE)
    pygame.display.set_caption(f"Rally sandbox - {params.name}")
    clock = pygame.time.Clock()
    camera = Camera(SCREEN_SIZE)
    renderer = Renderer(screen, camera)
    telemetry = TelemetryPanel(pygame.font.SysFont("monospace", 16))
    controls = InputHandler()

    state = initial_state(model, centreline)
    camera.snap(state[0, X], state[0, Y])
    accumulator = 0.0
    sim_time = 0.0
    running = True

    while running:
        frame_dt = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            controls.handle_event(event)
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    state = initial_state(model, centreline)
                    camera.snap(state[0, X], state[0, Y])
                    sim_time = 0.0
            elif event.type == pygame.MOUSEWHEEL:
                camera.zoom(1.1 ** event.y)

        # Fixed-step physics, decoupled from the frame rate.
        inputs = controls.read(frame_dt)
        accumulator += min(frame_dt, 0.25)
        steps = 0
        while accumulator >= PHYSICS_DT and steps < MAX_STEPS_PER_FRAME:
            state = model.step(state, inputs, PHYSICS_DT)
            accumulator -= PHYSICS_DT
            sim_time += PHYSICS_DT
            steps += 1
        if steps == MAX_STEPS_PER_FRAME:
            accumulator = 0.0

        camera.follow(state[0, X], state[0, Y], frame_dt)
        renderer.draw_background()
        renderer.draw_road(left, right, centreline)
        renderer.draw_cars(state)
        telemetry.draw(
            screen,
            [
                ("input", controls.source),
                ("speed", f"{state[0, SPEED] * 3.6:6.1f} km/h"),
                ("steer", f"{math.degrees(state[0, STEER]):6.1f} deg"),
                ("throttle", f"{inputs[0, THROTTLE_IN]:6.2f}"),
                ("brake", f"{inputs[0, BRAKE_IN]:6.2f}"),
                ("handbrake", f"{inputs[0, HANDBRAKE_IN]:6.2f}"),
                ("time", f"{sim_time:6.1f} s"),
                ("fps", f"{clock.get_fps():6.1f}"),
            ],
        )
        pygame.display.flip()

    pygame.quit()
