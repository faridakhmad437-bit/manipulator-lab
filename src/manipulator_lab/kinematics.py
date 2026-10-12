"""Forward kinematics, analytic inverse kinematics, and reachability.

Conventions (see SPEC section 3.1):
    * metres, radians; world frame at the shoulder, x right, y up, CCW positive
    * theta1 is absolute (from +x to link 1); theta2 is relative (link 2 w.r.t. link 1)
    * elbow-up  <=> theta2 <= 0,  elbow-down <=> theta2 >= 0
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .arm import PlanarArm

REACH_TOL = 1e-9
"""Absolute slack (metres) on the workspace radii; absorbs float noise only."""

BEYOND_MAX_REACH = "target_beyond_max_reach"
INSIDE_MIN_REACH = "target_inside_min_reach"


class Branch(Enum):
    """Which of the two IK solutions to return."""

    ELBOW_UP = "elbow_up"  # theta2 <= 0
    ELBOW_DOWN = "elbow_down"  # theta2 >= 0


@dataclass(frozen=True)
class IKResult:
    """Outcome of an analytic IK query. ``theta`` is None when unreachable."""

    reachable: bool
    theta: np.ndarray | None
    branch: Branch
    reason: str | None = None


def _vector2(value: object, name: str) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be two finite numbers.") from exc
    if arr.shape != (2,):
        raise ValueError(f"{name} must have exactly two values, got shape {arr.shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values.")
    return arr


def wrap_to_pi(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    wrapped = math.fmod(angle + math.pi, 2.0 * math.pi)
    if wrapped < 0.0:
        wrapped += 2.0 * math.pi
    return wrapped - math.pi


def fk(arm: PlanarArm, theta: object) -> np.ndarray:
    """End-effector position (x, y) for joint angles ``theta = (theta1, theta2)``."""
    t1, t2 = _vector2(theta, "theta")
    return np.array(
        [
            arm.l1 * math.cos(t1) + arm.l2 * math.cos(t1 + t2),
            arm.l1 * math.sin(t1) + arm.l2 * math.sin(t1 + t2),
        ]
    )


def _unreachable_reason(arm: PlanarArm, r: float, tol: float) -> str | None:
    if r > arm.max_reach + tol:
        return BEYOND_MAX_REACH
    if r < arm.min_reach - tol:
        return INSIDE_MIN_REACH
    return None


def is_reachable(arm: PlanarArm, target: object, *, tol: float = REACH_TOL) -> bool:
    """True when |l1 - l2| <= |target| <= l1 + l2, within ``tol`` metres."""
    x, y = _vector2(target, "target")
    return _unreachable_reason(arm, math.hypot(x, y), tol) is None


def ik_analytic(
    arm: PlanarArm,
    target: object,
    branch: Branch = Branch.ELBOW_UP,
    *,
    tol: float = REACH_TOL,
) -> IKResult:
    """Closed-form IK for the requested branch.

    Unreachable targets return ``reachable=False`` with a reason, never NaN.
    Within ``tol`` of a workspace boundary, cos(theta2) is clamped to [-1, 1]
    explicitly (the only place a clamp happens).

    When l1 == l2 and the target is the origin, theta1 is undefined (every theta1
    works); by convention theta1 = 0 is returned.
    """
    x, y = _vector2(target, "target")
    r = math.hypot(x, y)

    reason = _unreachable_reason(arm, r, tol)
    if reason is not None:
        return IKResult(reachable=False, theta=None, branch=branch, reason=reason)

    c2 = (r * r - arm.l1 * arm.l1 - arm.l2 * arm.l2) / (2.0 * arm.l1 * arm.l2)
    c2 = min(1.0, max(-1.0, c2))
    s2 = math.sqrt(max(0.0, 1.0 - c2 * c2))
    if branch is Branch.ELBOW_UP:
        s2 = -s2

    theta2 = math.atan2(s2, c2)
    theta1 = math.atan2(y, x) - math.atan2(arm.l2 * s2, arm.l1 + arm.l2 * c2)

    return IKResult(
        reachable=True,
        theta=np.array([wrap_to_pi(theta1), theta2]),
        branch=branch,
        reason=None,
    )
