"""Draws the world and cars with simple lines."""

from __future__ import annotations
import math
import numpy as np
import pygame
from rallysim.core.state import X, Y, YAW
from rallysim.viz.camera import Camera

BACKGROUND = (24, 26, 30)
GRID = (38, 41, 47)
ROAD_EDGE = (170, 140, 100)
ROAD_CENTRE = (80, 72, 60)
CAR = (230, 80, 60)
CAR_NOSE = (255, 220, 120)


class Renderer:
    def __init__(
            self,
            screen: pygame.Surface,
            camera: Camera,
            car_length: float = 4.0,
            car_width: float = 1.8,
            grid_spacing: float = 10.0,
    ) -> None:
        self.screen = screen
        self.camera = camera
        self.car_length = car_length
        self.car_width = car_width
        self.grid_spacing = grid_spacing

    def draw_background(self) -> None:
        self.screen.fill(BACKGROUND)
        cam, g = self.camera, self.grid_spacing
        half_w, half_h = cam.view_half_extent()

        x = math.floor((cam.cx - half_w) / g) * g
        while x <= cam.cx + half_w:
            pygame.draw.line(
                self.screen, GRID, cam.to_screen(x, cam.cy - half_h), cam.to_screen(x, cam.cy + half_h)
            )
            x += g
        y = math.floor((cam.cy - half_h) / g) * g
        while y <= cam.cy + half_h:
            pygame.draw.line(
                self.screen, GRID, cam.to_screen(cam.cx - half_w, y), cam.to_screen(cam.cx + half_w, y)
            )
            y += g

    def draw_road(self, left: np.ndarray, right: np.ndarray, centre: np.ndarray) -> None:
        cam = self.camera
        pygame.draw.lines(self.screen, ROAD_CENTRE, False, cam.to_screen_many(centre), 1)
        pygame.draw.lines(self.screen, ROAD_EDGE, False, cam.to_screen_many(left), 2)
        pygame.draw.lines(self.screen, ROAD_EDGE, False, cam.to_screen_many(right), 2)

    def draw_cars(self, state: np.ndarray) -> None:
        for row in state:
            self._draw_car(float(row[X]), float(row[Y]), float(row[YAW]))

    def _draw_car(self, x: float, y: float, yaw: float) -> None:
        c, s = math.cos(yaw), math.sin(yaw)
        hl, hw = self.car_length / 2, self.car_width / 2
        corners = [(hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw)]
        points = [self.camera.to_screen(x + px * c - py * s, y + px * s + py * c) for px, py in corners]
        pygame.draw.polygon(self.screen, CAR, points, 2)
        pygame.draw.line(
            self.screen, CAR_NOSE, self.camera.to_screen(x, y), self.camera.to_screen(x + hl * c, y + hl * s), 2
        )
