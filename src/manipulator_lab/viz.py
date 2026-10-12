"""Drawing and animation of the planar arm (needs matplotlib, a dev extra).

Not imported by ``manipulator_lab/__init__``: use ``from manipulator_lab import viz``.
Figures are built with ``matplotlib.figure.Figure`` directly, never through ``pyplot``, so
nothing here touches global pyplot state or needs a display.

The numerical part (``arm_points``, ``frame_positions``, ``frame_indices``) is plain NumPy and
is what the tests compare against forward kinematics; the drawing code only displays it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.figure import Figure
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError("manipulator_lab.viz needs matplotlib: pip install -e \".[dev]\"") from exc

from .arm import PlanarArm
from .kinematics import _vector2, fk
from .trajectory import PlanResult, Trajectory

# Reference palette (dataviz skill): light surface, text tokens, categorical slots 1-2.
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
TRAIL_COLOR, TARGET_COLOR = SERIES[0], SERIES[1]

_FIGSIZE = (4.6, 4.6)
_DPI = 100


# ------------------------------------------------------------- pure geometry
def arm_points(arm: PlanarArm, q: object) -> np.ndarray:
    """Base, elbow and tip as a (3, 2) array for joint angles ``q``."""
    q = _vector2(q, "q")
    elbow = (arm.l1 * math.cos(q[0]), arm.l1 * math.sin(q[0]))
    tip = np.asarray(fk(arm, q), dtype=float)
    return np.array([[0.0, 0.0], elbow, tip])


def frame_indices(n_samples: int, n_frames: int) -> np.ndarray:
    """``n_frames`` strictly increasing sample indices from 0 to n_samples - 1 inclusive."""
    for name, value in (("n_samples", n_samples), ("n_frames", n_frames)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer, got {value!r}.")
    if n_frames < 2 or n_samples < 2 or n_frames > n_samples:
        raise ValueError("need 2 <= n_frames <= n_samples.")
    idx = np.round(np.linspace(0, n_samples - 1, int(n_frames))).astype(int)
    if np.any(np.diff(idx) <= 0):  # cannot happen when n_frames <= n_samples; kept as a guard
        raise ValueError("frame indices are not strictly increasing.")
    return idx


def frame_positions(arm: PlanarArm, traj: Trajectory, indices: object = None) -> np.ndarray:
    """(n, 3, 2) arm points at the chosen samples of ``traj`` (default: every sample)."""
    idx = np.arange(len(traj.t)) if indices is None else np.asarray(indices, dtype=int)
    return np.array([arm_points(arm, traj.q[i]) for i in idx])


# --------------------------------------------------------------------- scene
def _positive_number(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a positive finite number, got {value!r}.")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}.")
    return value


def _check_plan(plan: object) -> None:
    if not isinstance(plan, PlanResult) or not plan.ok or plan.trajectory is None:
        raise ValueError("a successful PlanResult (plan.ok is True) is required.")


def _scene(arm: PlanarArm, title: str | None = None):
    fig = Figure(figsize=_FIGSIZE, dpi=_DPI, facecolor=SURFACE)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_SECONDARY)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=3)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    angle = np.linspace(-math.pi, math.pi, 361)
    ax.plot(arm.max_reach * np.cos(angle), arm.max_reach * np.sin(angle), color=TEXT_SECONDARY, lw=0.8)
    ax.plot(arm.min_reach * np.cos(angle), arm.min_reach * np.sin(angle), color=TEXT_SECONDARY, lw=0.8, ls="--")
    ax.plot([0.0], [0.0], "s", color=TEXT_PRIMARY, ms=6, zorder=5)
    lim = 1.08 * arm.max_reach
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)", color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel("y (m)", color=TEXT_SECONDARY, fontsize=9)
    if title:
        ax.set_title(title, fontsize=10, color=TEXT_PRIMARY, loc="left")
    return fig, ax


def _mark_target(ax, target: np.ndarray, label: str | None = "target"):
    ax.plot([target[0]], [target[1]], "x", color=TARGET_COLOR, ms=10, mew=2.2, zorder=6, label=label)


def _save_png(fig, path) -> None:
    fig.savefig(Path(path), dpi=150, facecolor=SURFACE)


# ----------------------------------------------------------------- animation
@dataclass
class ArmAnimation:
    """Drawn scene plus the data it shows; ``set_frame(i)`` moves the artists to frame ``i``."""

    fig: object
    positions: np.ndarray
    times: np.ndarray
    arm_line: object
    trail_line: object
    time_text: object

    def set_frame(self, i: int) -> None:
        if isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer)):
            raise ValueError(f"frame index must be an integer, got {i!r}.")
        if not 0 <= i < len(self.positions):
            raise ValueError(f"frame index {i} outside [0, {len(self.positions) - 1}].")
        pts = self.positions[i]
        self.arm_line.set_data(pts[:, 0], pts[:, 1])
        self.trail_line.set_data(self.positions[: i + 1, 2, 0], self.positions[: i + 1, 2, 1])
        self.time_text.set_text(f"t = {self.times[i]:.2f} s")


def animate_plan(
    arm: PlanarArm, plan: PlanResult, target: object, *, n_frames: int = 60, title: str | None = None
) -> ArmAnimation:
    """Scene for a successful plan: arm, growing end-effector trail, target marker, clock."""
    _check_plan(plan)
    goal = _vector2(target, "target")
    traj = plan.trajectory
    idx = frame_indices(len(traj.t), n_frames)
    positions = frame_positions(arm, traj, idx)

    fig, ax = _scene(arm, title)
    ax.plot(positions[:, 2, 0], positions[:, 2, 1], color=TEXT_SECONDARY, lw=0.9, ls=":", zorder=2)
    (trail,) = ax.plot([], [], color=TRAIL_COLOR, lw=2.0, zorder=3, label="end-effector path")
    (arm_line,) = ax.plot([], [], "-o", color=TEXT_PRIMARY, lw=3.0, ms=5, zorder=4)
    _mark_target(ax, goal)
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY, loc="lower left")
    clock = ax.text(0.98, 0.02, "", transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=9, color=TEXT_SECONDARY)
    anim = ArmAnimation(fig=fig, positions=positions, times=traj.t[idx], arm_line=arm_line,
                        trail_line=trail, time_text=clock)
    anim.set_frame(0)
    return anim


def save_gif(anim: ArmAnimation, path, fps: float = 20, hold: int = 0) -> None:
    """Write the animation as a GIF (Pillow writer; no ffmpeg needed).

    ``hold`` repeats the last frame that many extra times; the encoder merges identical
    frames, so this lengthens the pause on the final pose rather than adding frames.
    """
    path = Path(path)
    if path.suffix.lower() != ".gif":
        raise ValueError(f"path must end in .gif, got {path.name!r}.")
    fps = _positive_number(fps, "fps")
    if isinstance(hold, (bool, np.bool_)) or not isinstance(hold, (int, np.integer)) or hold < 0:
        raise ValueError(f"hold must be a non-negative integer, got {hold!r}.")
    last = len(anim.positions) - 1
    frames = list(range(last + 1)) + [last] * int(hold)
    runner = FuncAnimation(anim.fig, anim.set_frame, frames=frames, repeat=False)
    runner.save(path, writer=PillowWriter(fps=fps))
    anim.set_frame(0)


# ------------------------------------------------------------------ figures
def plot_snapshots(arm: PlanarArm, plan: PlanResult, target: object, path, n: int = 6):
    """Static figure: ``n`` arm poses along the plan, fading in towards the last one."""
    _check_plan(plan)
    goal = _vector2(target, "target")
    traj = plan.trajectory
    idx = frame_indices(len(traj.t), n)
    positions = frame_positions(arm, traj, idx)
    fig, ax = _scene(arm, "Arm poses along the trajectory")
    full = frame_positions(arm, traj)
    ax.plot(full[:, 2, 0], full[:, 2, 1], color=TRAIL_COLOR, lw=1.6, zorder=2, label="end-effector path")
    for k, pts in enumerate(positions):
        alpha = 0.18 + 0.82 * k / (len(positions) - 1)
        ax.plot(pts[:, 0], pts[:, 1], "-o", color=TEXT_PRIMARY, lw=2.4, ms=4, alpha=alpha, zorder=4)
    _mark_target(ax, goal)
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY, loc="lower left")
    fig.tight_layout()
    _save_png(fig, path)
    return fig


def plot_unreachable(arm: PlanarArm, q0: object, plan: PlanResult, target: object, path):
    """Static figure for a target the planner rejected: the start pose, the target, the reason."""
    if not isinstance(plan, PlanResult) or plan.ok:
        raise ValueError("a failed PlanResult (plan.ok is False) is required.")
    goal = _vector2(target, "target")
    reason = str(plan.reason).replace("_", " ")
    radius = math.hypot(*goal)
    if radius > arm.max_reach:
        detail = f"{radius:.2f} m from the shoulder; max reach {arm.max_reach:.2f} m"
    else:
        detail = f"{radius:.2f} m from the shoulder; min reach {arm.min_reach:.2f} m"
    fig, ax = _scene(arm, f"Unreachable target: {reason}\n{detail}")
    pts = arm_points(arm, q0)
    ax.plot(pts[:, 0], pts[:, 1], "-o", color=TEXT_PRIMARY, lw=3.0, ms=5, zorder=4)
    ax.plot([0.0, goal[0]], [0.0, goal[1]], color=TEXT_SECONDARY, lw=0.9, ls=":", zorder=2)
    _mark_target(ax, goal, None)
    ax.annotate("target (no trajectory planned)", xy=goal, xytext=(-12, 0), textcoords="offset points",
                ha="right", va="center", fontsize=8, color=TEXT_SECONDARY, zorder=7,
                bbox={"fc": SURFACE, "ec": "none", "pad": 1.5})
    fig.tight_layout()
    _save_png(fig, path)
    return fig
