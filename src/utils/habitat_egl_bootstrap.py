"""Headless EGL / NVIDIA env for Habitat-Sim in Docker and bare-metal."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def _apply_shell_env(script: Path) -> None:
    if not script.is_file():
        return
    out = subprocess.check_output(
        ["bash", "-c", f"source '{script}' && env -0"],
        stderr=subprocess.DEVNULL,
    )
    for chunk in out.split(b"\0"):
        if not chunk or b"=" not in chunk:
            continue
        key, _, val = chunk.partition(b"=")
        os.environ[key.decode(errors="replace")] = val.decode(errors="replace")


def bootstrap_habitat_egl(*, strict: bool = False) -> bool:
    """
    Configure NVIDIA/Mesa EGL similarly to ``docker/ensure_habitat_egl.sh``.

    Returns True if bootstrap script ran without error (or scripts are absent).
    """
    egl_bootstrap = _ROOT / "docker" / "ensure_habitat_egl.sh"
    nv_env = _ROOT / "docker" / "nvidia_habitat_env.sh"

    ok = True
    if egl_bootstrap.is_file():
        proc = subprocess.run(["bash", str(egl_bootstrap)], check=False)
        ok = proc.returncode == 0
        if not ok and strict:
            print(f"[habitat-egl] FAIL: {egl_bootstrap}", file=sys.stderr)
            return False

    try:
        _apply_shell_env(nv_env)
    except subprocess.CalledProcessError:
        if strict:
            print(f"[habitat-egl] FAIL: source {nv_env}", file=sys.stderr)
            return False

    # Habitat headless: avoid X11; keep NVIDIA visible for EGL in containers.
    os.environ.setdefault("NVIDIA_VISIBLE_DEVICES", "all")
    os.environ.setdefault("NVIDIA_DRIVER_CAPABILITIES", "all")
    os.environ.pop("DISPLAY", None)
    return ok
