"""Command-line entry point: `rally <command> [options]`."""

from __future__ import annotations
import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="rally", description="Rally simulation tools")
    sub = parser.add_subparsers(dest="command", required=True)

    sandbox = sub.add_parser("sandbox", help="drive the car yourself on a test road")
    sandbox.add_argument("--car", type=Path, default=None, help="car config file")

    sub.add_parser("gamepad-test", help="show gamepad axis values")

    launch = sub.add_parser("launch", help="start several processes from a launch file")
    launch.add_argument("file", type=Path, help="YAML launch file")

    args = parser.parse_args(argv)

    # Imports inside each branch, so e.g. `rally launch` doesn't load pygame.
    if args.command == "sandbox":
        from rallysim.apps.sandbox import run

        run(car=args.car)
    elif args.command == "gamepad-test":
        from rallysim.apps.gamepad_test import run

        run()
    elif args.command == "launch":
        from rallysim.apps.launch import run

        run(args.file)


if __name__ == "__main__":
    main()
