"""Joint-space trajectories from q0 to qf in time T (SPEC section 3.5).

* ``linear``: constant velocity; velocity jumps from rest at t = 0 and back at t = T.
* ``cubic``: rest-to-rest (qd(0) = qd(T) = 0), q = q0 + delta (3 s^2 - 2 s^3), s = t / T.

The joint difference is ``qf - q0`` taken literally (no shortest-way angle wrapping), so the
motion stays inside the range spanned by the two given configurations.

The cubic coefficients are available two independent ways: ``cubic_coefficients`` (closed
form) and ``cubic_coefficients_via_linear_system`` (the 4x4 boundary-condition system solved
with NMT's ``gaussian_elimination``). The second exists only as a cross-check; it is the one
use of NMT outside ``ik_numeric.py`` (SPEC section 4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numerical_toolkit import gaussian_elimination

from .arm import PlanarArm
from .kinematics import Branch, _vector2, ik_analytic

LINEAR = "linear"
CUBIC = "cubic"
PROFILES = (LINEAR, CUBIC)


@dataclass(frozen=True)
class Trajectory:
    """Joint trajectory sampled at times ``t``; ``q``, ``qd``, ``qdd`` have shape (len(t), 2).

    ``qdd`` is the acceleration strictly inside (0, T); the velocity jump of the linear
    profile at the ends is an impulse and is not represented in it.
    """

    t: np.ndarray
    q: np.ndarray
    qd: np.ndarray
    qdd: np.ndarray
    profile: str


@dataclass(frozen=True)
class PlanResult:
    """Outcome of ``plan_to_target``. ``trajectory`` and ``theta_target`` are None on failure."""

    ok: bool
    trajectory: Trajectory | None
    theta_target: np.ndarray | None
    branch: Branch
    reason: str | None = None


def _duration(value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"duration T must be a real number, got {value!r}.")
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"duration T must be finite and positive, got {value!r}.")
    return value


def _times(value: object, duration: float) -> np.ndarray:
    try:
        t = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("time samples must be real numbers.") from exc
    if t.ndim != 1 or t.size == 0:
        raise ValueError("time samples must be a non-empty 1-D sequence.")
    if not np.all(np.isfinite(t)):
        raise ValueError("time samples must be finite.")
    if np.any(t < 0.0) or np.any(t > duration):
        raise ValueError(f"time samples must lie in [0, T] = [0, {duration}].")
    return t


def _scalar(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite real number.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real number.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite real number.")
    return number


def time_grid(duration: float, n: int) -> np.ndarray:
    """``n >= 2`` equally spaced times from exactly 0 to exactly T."""
    duration = _duration(duration)
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 2:
        raise ValueError(f"n must be an integer >= 2, got {n!r}.")
    return np.linspace(0.0, duration, int(n))


def linear(q0: object, qf: object, duration: float, t: object) -> Trajectory:
    """Constant-velocity interpolation from ``q0`` to ``qf``."""
    start, end = _vector2(q0, "q0"), _vector2(qf, "qf")
    duration = _duration(duration)
    t = _times(t, duration)
    delta = end - start
    s = (t / duration)[:, None]
    q = start + delta * s
    qd = np.broadcast_to(delta / duration, q.shape).copy()
    qdd = np.zeros_like(q)
    return Trajectory(t=t, q=q, qd=qd, qdd=qdd, profile=LINEAR)


def cubic(q0: object, qf: object, duration: float, t: object) -> Trajectory:
    """Rest-to-rest cubic from ``q0`` to ``qf`` (closed form)."""
    start, end = _vector2(q0, "q0"), _vector2(qf, "qf")
    duration = _duration(duration)
    t = _times(t, duration)
    delta = end - start
    s = (t / duration)[:, None]
    q = start + delta * (3.0 * s**2 - 2.0 * s**3)
    qd = delta * (6.0 * s - 6.0 * s**2) / duration
    qdd = delta * (6.0 - 12.0 * s) / duration**2
    return Trajectory(t=t, q=q, qd=qd, qdd=qdd, profile=CUBIC)


def cubic_coefficients(q0: float, qf: float, duration: float) -> np.ndarray:
    """(a0, a1, a2, a3) of q(t) = a0 + a1 t + a2 t^2 + a3 t^3 for one joint, closed form."""
    q0, qf = _scalar(q0, "q0"), _scalar(qf, "qf")
    duration = _duration(duration)
    delta = qf - q0
    return np.array([q0, 0.0, 3.0 * delta / duration**2, -2.0 * delta / duration**3])


def cubic_coefficients_via_linear_system(q0: float, qf: float, duration: float) -> np.ndarray:
    """Same coefficients from the 4x4 boundary system, solved with NMT ``gaussian_elimination``.

    Rows: q(0) = q0, qd(0) = 0, q(T) = qf, qd(T) = 0.
    """
    q0, qf = _scalar(q0, "q0"), _scalar(qf, "qf")
    T = _duration(duration)
    matrix = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [1.0, T, T**2, T**3],
            [0.0, 1.0, 2.0 * T, 3.0 * T**2],
        ]
    )
    rhs = np.array([q0, 0.0, qf, 0.0])
    return np.asarray(gaussian_elimination(matrix, rhs), dtype=float)


def plan_to_target(
    arm: PlanarArm,
    q0: object,
    target: object,
    duration: float,
    t: object,
    *,
    profile: str = CUBIC,
    branch: Branch = Branch.ELBOW_UP,
) -> PlanResult:
    """Move from joint configuration ``q0`` to the analytic IK solution for ``target``.

    The goal configuration is on ``branch`` (default elbow-up, theta2 <= 0). An unreachable
    target returns ``ok=False`` with the reason from ``ik_analytic`` and no trajectory.
    Invalid arguments raise ``ValueError``.
    """
    if profile not in PROFILES:
        raise ValueError(f"profile must be one of {PROFILES}, got {profile!r}.")
    if not isinstance(branch, Branch):
        raise ValueError(f"branch must be a Branch, got {branch!r}.")
    start = _vector2(q0, "q0")
    goal = _vector2(target, "target")
    duration = _duration(duration)
    t = _times(t, duration)

    ik = ik_analytic(arm, goal, branch)
    if not ik.reachable:
        return PlanResult(ok=False, trajectory=None, theta_target=None, branch=branch, reason=ik.reason)
    build = linear if profile == LINEAR else cubic
    return PlanResult(
        ok=True,
        trajectory=build(start, ik.theta, duration, t),
        theta_target=np.asarray(ik.theta, dtype=float),
        branch=branch,
    )
