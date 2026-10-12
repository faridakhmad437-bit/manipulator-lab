"""Manipulator Lab: planar 2-link kinematics and trajectory planning."""

from .arm import PlanarArm
from .kinematics import (
    BEYOND_MAX_REACH,
    INSIDE_MIN_REACH,
    REACH_TOL,
    Branch,
    IKResult,
    fk,
    ik_analytic,
    is_reachable,
    wrap_to_pi,
)

__all__ = [
    "BEYOND_MAX_REACH",
    "INSIDE_MIN_REACH",
    "REACH_TOL",
    "Branch",
    "IKResult",
    "PlanarArm",
    "fk",
    "ik_analytic",
    "is_reachable",
    "wrap_to_pi",
]
__version__ = "0.1.0"
