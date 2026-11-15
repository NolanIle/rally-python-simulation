"""Start several processes from a YAML launch file and stop them together.

Launch file format:
    processes:
      - name: sandbox
        cmd: rally sandbox
If any process exits, or on Ctrl+C, all processes are stopped.
"""

from __future__ import annotations
import shlex
import subprocess
import time
from pathlib import Path
import yaml


def run(path: Path) -> None:
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    entries = spec.get("processes", []) if isinstance(spec, dict) else []
    if not entries:
        raise SystemExit(f"{path}: no processes defined")

    procs: list[tuple[str, subprocess.Popen]] = []
    try:
        for entry in entries:
            name, cmd = entry["name"], entry["cmd"]
            print(f"[launch] starting {name}: {cmd}")
            procs.append((name, subprocess.Popen(shlex.split(cmd))))

        while True:
            for name, proc in procs:
                code = proc.poll()
                if code is not None:
                    print(f"[launch] {name} exited with code {code}, stopping all")
                    return
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\n[launch] stopping all")
    finally:
        for _, proc in procs:
            if proc.poll() is None:
                proc.terminate()
        for _, proc in procs:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
