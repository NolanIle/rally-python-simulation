"""Print gamepad axis values, to find the right axis numbers for GamepadMapping."""

from __future__ import annotations

import pygame


def run() -> None:
    pygame.init()
    pygame.display.set_mode((300, 100))
    pygame.display.set_caption("Gamepad test - move sticks and triggers")
    clock = pygame.time.Clock()
    joystick = None
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.JOYDEVICEADDED and joystick is None:
                joystick = pygame.joystick.Joystick(event.device_index)
                print(f"Connected: {joystick.get_name()}")
        if joystick is not None:
            axes = [f"{i}:{joystick.get_axis(i):+.2f}" for i in range(joystick.get_numaxes())]
            print("  ".join(axes), end="\r", flush=True)
        clock.tick(20)
    pygame.quit()
