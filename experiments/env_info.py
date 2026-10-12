"""Collect the measurement environment (SPEC section 7) so every result file carries it."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path


def _cpu_name() -> str:
    """Best-effort CPU model; falls back to platform.processor()."""
    try:
        if sys.platform == "win32":
            out = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)",
                ],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            ).stdout.strip()
            if out:
                return out
        elif sys.platform.startswith("linux"):
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        elif sys.platform == "darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            ).stdout.strip()
            if out:
                return out
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or "unknown"


def _git_commit(repo: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def collect() -> dict:
    """Environment metadata as a JSON-serialisable dict."""
    import numpy

    import manipulator_lab

    info = {
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": _cpu_name(),
        "logical_cpus": os.cpu_count(),
        "numpy": numpy.__version__,
        "manipulator_lab": manipulator_lab.__version__,
        "git_commit": _git_commit(Path(__file__).resolve().parent.parent),
    }
    try:
        import numerical_toolkit

        info["numerical_toolkit"] = numerical_toolkit.__version__
    except ImportError:
        info["numerical_toolkit"] = None
    return info


if __name__ == "__main__":
    for key, value in collect().items():
        print(f"{key}: {value}")
