"""Driving sandbox: drive the car yourself on a hand-made test road.

Controls: gamepad (left stick, triggers) or arrows / WASD.
R resets, mouse wheel zooms, Esc quits.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pygame

from rallysim.core.config import load_car, load_sim
from rallysim.core.state import (
    BRAKE_IN,
    HANDBRAKE_IN,
    SPEED,
    STEER,
    THROTTLE_IN,
    X,
    Y,
    YAW,
    YAW_RATE,
    body_slip,
    make_state,
)
from rallysim.physics.model import VehicleModel
from rallysim.physics.registry import build_model
from rallysim.viz.camera import Camera
from rallysim.viz.input import InputHandler
from rallysim.viz.renderer import Renderer
from rallysim.viz.telemetry import TelemetryPanel
from rallysim.world.test_road import make_test_road, road_edges

# src/rallysim/apps/sandbox.py -> repository root is three levels up
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CAR = ROOT / "config" / "cars" / "lancia_delta_s4.yaml"
DEFAULT_SIM = ROOT / "config" / "sim" / "default.yaml"

MAX_STEPS_PER_FRAME = 20  # avoid a spiral of death if a frame is very slow
SCREEN_SIZE = (1280, 800)
ROAD_WIDTH = 7.0  # [m]


def initial_state(model: VehicleModel, centreline: np.ndarray) -> np.ndarray:
    state = make_state(1, model.state_size)
    state[0, X], state[0, Y] = centreline[0]
    direction = centreline[1] - centreline[0]
    state[0, YAW] = math.atan2(direction[1], direction[0])
    return state


def run(car: Path | None = None, sim: Path | None = None) -> None:
    params = load_car(car or DEFAULT_CAR)
    sim_params = load_sim(sim or DEFAULT_SIM)
    model = build_model(sim_params, params)
    physics_dt = sim_params.physics_dt
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
                camera.zoom(1.1**event.y)

        # Fixed-step physics, decoupled from the frame rate.
        inputs = controls.read(frame_dt)
        accumulator += min(frame_dt, 0.25)
        steps = 0
        while accumulator >= physics_dt and steps < MAX_STEPS_PER_FRAME:
            state = model.step(state, inputs, physics_dt)
            accumulator -= physics_dt
            sim_time += physics_dt
            steps += 1
        if steps == MAX_STEPS_PER_FRAME:
            accumulator = 0.0

        camera.follow(state[0, X], state[0, Y], frame_dt)
        renderer.draw_background()
        renderer.draw_road(left, right, centreline)
        renderer.draw_cars(state)
        lines = [
            ("input", controls.source),
            ("model", sim_params.model),
            ("speed", f"{state[0, SPEED] * 3.6:6.1f} km/h"),
            ("steer", f"{math.degrees(state[0, STEER]):6.1f} deg"),
            ("throttle", f"{inputs[0, THROTTLE_IN]:6.2f}"),
            ("brake", f"{inputs[0, BRAKE_IN]:6.2f}"),
            ("handbrake", f"{inputs[0, HANDBRAKE_IN]:6.2f}"),
        ]
        # Only models that let the car slide carry these, so show them when
        # present rather than assuming the state layout.
        if model.state_size > YAW_RATE:
            lines.extend(
                [
                    ("slip", f"{math.degrees(body_slip(state[0])):6.1f} deg"),
                    ("yaw rate", f"{math.degrees(state[0, YAW_RATE]):6.1f} deg/s"),
                ]
            )
        lines.extend([("time", f"{sim_time:6.1f} s"), ("fps", f"{clock.get_fps():6.1f}")])
        telemetry.draw(screen, lines)
        pygame.display.flip()

    pygame.quit()
