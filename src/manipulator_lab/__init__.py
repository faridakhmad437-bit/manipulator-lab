"""Manipulator Lab: planar 2-link kinematics and trajectory planning."""

from . import trajectory
from .arm import PlanarArm
from .kinematics import (
    BEYOND_MAX_REACH,
    INSIDE_MIN_REACH,
    REACH_TOL,
    SINGULARITY_TOL,
    Branch,
    IKResult,
    condition_from_theta2,
    condition_number,
    fk,
    ik_analytic,
    is_reachable,
    is_singular,
    jacobian,
    wrap_to_pi,
)
from .ik_numeric import (
    MAX_ITERATIONS,
    NON_FINITE_VALUE,
    SINGULAR_JACOBIAN,
    NumericIKResult,
    default_initial_guess,
    ik_newton,
)
from .trajectory import PlanResult, Trajectory
from .workspace import ConditionMap, condition_map, workspace_boundary

__all__ = [
    "BEYOND_MAX_REACH",
    "INSIDE_MIN_REACH",
    "MAX_ITERATIONS",
    "NON_FINITE_VALUE",
    "REACH_TOL",
    "SINGULAR_JACOBIAN",
    "SINGULARITY_TOL",
    "Branch",
    "ConditionMap",
    "IKResult",
    "NumericIKResult",
    "PlanResult",
    "PlanarArm",
    "Trajectory",
    "condition_from_theta2",
    "condition_map",
    "condition_number",
    "default_initial_guess",
    "fk",
    "ik_analytic",
    "ik_newton",
    "is_reachable",
    "is_singular",
    "jacobian",
    "trajectory",
    "wrap_to_pi",
    "workspace_boundary",
]
__version__ = "0.5.0"
