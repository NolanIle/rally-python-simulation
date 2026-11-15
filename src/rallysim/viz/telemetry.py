"""Live telemetry panel drawn as text."""

from __future__ import annotations
import pygame

TEXT = (220, 220, 220)


class TelemetryPanel:
    def __init__(
            self, font: pygame.font.Font, origin: tuple[int, int] = (12, 12), line_height: int = 20
    ) -> None:
        self.font = font
        self.origin = origin
        self.line_height = line_height

    def draw(self, surface: pygame.Surface, lines: list[tuple[str, str]]) -> None:
        x, y = self.origin
        for label, value in lines:
            text = self.font.render(f"{label:<10}{value}", True, TEXT)
            surface.blit(text, (x, y))
            y += self.line_height
