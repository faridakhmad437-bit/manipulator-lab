"""Workspace boundary and conditioning map of the planar 2-link arm."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from .arm import PlanarArm
from .kinematics import Branch, condition_number, ik_analytic


@dataclass(frozen=True)
class ConditionMap:
    """cond(J) sampled on a square (x, y) grid.

    ``cond[iy, ix]`` belongs to the point (``x[ix]``, ``y[iy]``). Cells whose point is
    unreachable are NaN in ``cond`` and False in ``reachable``; this is an explicit
    mask, so use ``reachable`` rather than testing for NaN.
    """

    x: np.ndarray
    y: np.ndarray
    cond: np.ndarray
    reachable: np.ndarray


def _integer_at_least(value: object, minimum: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}.")
    return int(value)


def workspace_boundary(arm: PlanarArm, n: int = 361) -> tuple[np.ndarray, np.ndarray]:
    """Outer and inner workspace circles as closed (n, 2) polylines."""
    n = _integer_at_least(n, 3, "n")
    angle = np.linspace(-math.pi, math.pi, n)
    circle = np.column_stack([np.cos(angle), np.sin(angle)])
    return arm.max_reach * circle, arm.min_reach * circle


def condition_map(
    arm: PlanarArm,
    resolution: int = 201,
    *,
    extent: float | None = None,
    branch: Branch = Branch.ELBOW_UP,
) -> ConditionMap:
    """Sample cond(J) at the IK solution of every grid point.

    The grid spans [-extent, extent] in both axes (default 1.1 * max reach).
    cond(J) depends only on theta2, so the result is the same for both branches and
    radially symmetric; the map is still computed point by point through the public
    IK and Jacobian functions so it exercises the same code paths as the rest of the
    library.
    """
    resolution = _integer_at_least(resolution, 2, "resolution")
    if extent is None:
        extent = 1.1 * arm.max_reach
    elif isinstance(extent, bool) or not isinstance(extent, Real) or not (
        math.isfinite(extent) and extent > 0.0
    ):
        raise ValueError(f"extent must be a positive finite number, got {extent!r}.")

    axis = np.linspace(-extent, extent, resolution)
    cond = np.full((resolution, resolution), np.nan)
    reachable = np.zeros((resolution, resolution), dtype=bool)
    for iy, y in enumerate(axis):
        for ix, x in enumerate(axis):
            result = ik_analytic(arm, (x, y), branch)
            if result.reachable:
                reachable[iy, ix] = True
                cond[iy, ix] = condition_number(arm, result.theta)
    return ConditionMap(x=axis.copy(), y=axis.copy(), cond=cond, reachable=reachable)
