# rallysim

A rally driving simulation sandbox. Vehicle physics in Python, real-time
visualisation with pygame-ce.

Status: early. The vehicle model is a kinematic bicycle model with grip limits —
a placeholder until a proper dynamic model with tire forces exists. See
[TODO.md](TODO.md).

## Requirements

- Linux (developed on Ubuntu 24.04)
- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/) for dependency management

Install uv if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Setup

```bash
uv sync --extra dev
```

That creates `.venv/`, installs everything from `uv.lock`, and installs
`rallysim` itself in editable mode — edits under `src/` take effect immediately
with no reinstall.

Dependency versions are pinned in `uv.lock`, so this gives the same environment
on any machine. There is no need to activate the venv manually; `uv run` handles
it.

## Important: running alongside ROS 2

**If you have ROS 2 sourced in your shell, prefix every command with
`env -u PYTHONPATH`.**

```bash
env -u PYTHONPATH uv run rally sandbox
```

Sourcing ROS 2 sets `PYTHONPATH` to its own `site-packages`. Python prepends
`PYTHONPATH` to `sys.path` for *every* interpreter, including one inside a
virtualenv — so ROS packages leak into this project and pytest tries to load
ROS's pytest plugins, which fails. `PYTHONPATH` is read before a venv's
`include-system-site-packages = false` is applied, so there is no venv-level
setting that prevents this; it has to be cleared per command.

`env -u PYTHONPATH` removes the variable for that one process only. Your shell
keeps its ROS setup and ROS keeps working normally in the same terminal.

An alias makes it less tedious — add to `~/.bashrc`:

```bash
alias rallyrun='env -u PYTHONPATH uv run'
```

Then `rallyrun rally sandbox`, `rallyrun pytest`, and so on. The rest of this
README writes commands as plain `uv run ...`; add the prefix if ROS is sourced.

## Running

### Drive the car

```bash
uv run rally sandbox
```

Drive on a flat test road. Optionally pick a car config:

```bash
uv run rally sandbox --car config/cars/lancia_delta_s4.yaml
```

**Keyboard**

| Key | Action |
| --- | --- |
| `↑` / `W` | Throttle |
| `↓` / `S` | Brake |
| `←` / `A` | Steer left |
| `→` / `D` | Steer right |
| `Space` | Handbrake |

**Gamepad** — plugged-in controllers are picked up automatically. Defaults are
left stick for steering, right trigger throttle, left trigger brake, and the
bottom face button for handbrake. Axis and button numbering varies between
controllers; check yours with:

```bash
uv run rally gamepad-test
```

### Launch several processes

```bash
uv run rally launch config/launch/sandbox.yaml
```

Starts the processes listed in a YAML launch file.

## Development

```bash
uv run pytest                  # tests
uv run ruff check .            # lint
uv run ruff format .           # format
```

Managing dependencies:

```bash
uv add <package>               # add a runtime dependency
uv add --optional dev <pkg>    # add a dev dependency
uv remove <package>            # remove one
uv sync --extra dev            # rebuild the env from uv.lock
uv lock --upgrade              # re-resolve to newer versions
```

`uv add` / `uv remove` update both `pyproject.toml` and `uv.lock`. Commit
`uv.lock` — it is what makes builds reproducible.

## Layout

```
src/rallysim/
  cli.py          `rally` entry point
  apps/           runnable apps: sandbox, gamepad-test, launch
  core/           config loading, vehicle state
  physics/        vehicle models
  viz/            rendering, camera, input, telemetry overlay
  world/          test track
config/cars/      vehicle parameter files
config/launch/    launch files
tests/            pytest suite
```