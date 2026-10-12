"""Numerical IK: adapter around ``numerical_toolkit.newton_system`` (NMT, pinned v1.0.0).

This is the only module that imports NMT for IK (SPEC section 4), so the dependency stays
easy to pin and to replace. Facts about NMT behaviour that this adapter handles
(SPEC section 4, verified by reading the source and by smoke tests):

* the branch Newton converges to is decided by the initial guess;
* NMT's default initial guess (0, 0) is exactly singular and makes it raise;
* ``newton_system`` raises (instead of returning) when J is exactly singular or when a
  non-finite value appears, so those cases are converted into structured results here;
* NMT reports iterations only in its result, not when it raises.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numerical_toolkit import (
    NonFiniteValueError,
    NumericalMethodError,
    SolverOptions,
    newton_system,
)

from .arm import PlanarArm
from .kinematics import (
    Branch,
    _unreachable_reason,
    _vector2,
    REACH_TOL,
    fk,
    jacobian,
    wrap_to_pi,
)

MAX_ITERATIONS = "max_iterations"
SINGULAR_JACOBIAN = "singular_jacobian"
NON_FINITE_VALUE = "non_finite_value"

_BRANCH_AMBIGUITY_TOL = 1e-9
"""|sin(theta2)| below this: both analytic branches coincide, so no branch is reported."""


@dataclass(frozen=True)
class NumericIKResult:
    """Outcome of a Newton IK run.

    ``theta`` is wrapped to [-pi, pi]; it is the converged solution, or the last iterate when
    the iteration limit was hit, or None when nothing was attempted or the solver aborted.
    ``iterations`` is None when the solver aborted with an exception (NMT does not report
    a count in that case). ``reason`` is None exactly when ``converged`` is True; otherwise
    it is ``BEYOND_MAX_REACH``, ``INSIDE_MIN_REACH``, ``MAX_ITERATIONS``,
    ``SINGULAR_JACOBIAN`` or ``NON_FINITE_VALUE``.
    """

    converged: bool
    theta: np.ndarray | None
    iterations: int | None
    residual: float | None
    step_error: float | None
    reason: str | None
    message: str
    branch: Branch | None
    history: tuple | None = None


def _check_branch(branch: object) -> Branch:
    if not isinstance(branch, Branch):
        raise ValueError(f"branch must be a Branch, got {branch!r}.")
    return branch


def default_initial_guess(
    arm: PlanarArm, target: object, branch: Branch = Branch.ELBOW_UP
) -> np.ndarray:
    """Regular (non-singular) starting point on the requested side.

    theta2 = -pi/2 (elbow-up) or +pi/2 (elbow-down); theta1 points link 1 so that the
    end effector starts on the target's side of the shoulder.
    """
    x, y = _vector2(target, "target")
    branch = _check_branch(branch)
    t2 = -math.pi / 2.0 if branch is Branch.ELBOW_UP else math.pi / 2.0
    t1 = math.atan2(y, x) - math.atan2(arm.l2 * math.sin(t2), arm.l1 + arm.l2 * math.cos(t2))
    return np.array([wrap_to_pi(t1), t2])


def _branch_of(theta: np.ndarray) -> Branch | None:
    if abs(math.sin(theta[1])) < _BRANCH_AMBIGUITY_TOL:
        return None
    return Branch.ELBOW_UP if theta[1] < 0.0 else Branch.ELBOW_DOWN


def _failure(reason: str, message: str, iterations: int | None) -> NumericIKResult:
    return NumericIKResult(
        converged=False,
        theta=None,
        iterations=iterations,
        residual=None,
        step_error=None,
        reason=reason,
        message=message,
        branch=None,
    )


def ik_newton(
    arm: PlanarArm,
    target: object,
    initial: object | None = None,
    *,
    branch: Branch = Branch.ELBOW_UP,
    options: SolverOptions | None = None,
) -> NumericIKResult:
    """Solve IK with Newton's method on F(q) = fk(q) - target using this library's Jacobian.

    ``initial`` defaults to ``default_initial_guess(arm, target, branch)``; ``branch`` is
    used only for that default. Unreachable targets are not attempted. Invalid arguments
    raise ``ValueError``; numerical failures are returned, never raised.
    """
    goal = _vector2(target, "target")
    branch = _check_branch(branch)
    if options is not None and not isinstance(options, SolverOptions):
        raise ValueError("options must be a SolverOptions instance or None.")
    start = (
        default_initial_guess(arm, goal, branch)
        if initial is None
        else _vector2(initial, "initial")
    )

    unreachable = _unreachable_reason(arm, math.hypot(*goal), REACH_TOL)
    if unreachable is not None:
        return _failure(unreachable, "Target lies outside the reachable workspace.", 0)

    def residual_fn(q):
        return fk(arm, q) - goal

    def jacobian_fn(q):
        return jacobian(arm, q)

    try:
        nmt = newton_system(residual_fn, jacobian_fn, start, options=options)
    except NonFiniteValueError as exc:
        return _failure(NON_FINITE_VALUE, str(exc), None)
    except NumericalMethodError as exc:
        if "singular" in str(exc).lower():
            return _failure(SINGULAR_JACOBIAN, str(exc), None)
        raise

    theta = np.array([wrap_to_pi(float(v)) for v in nmt.solution])
    return NumericIKResult(
        converged=nmt.converged,
        theta=theta,
        iterations=nmt.iterations,
        residual=nmt.residual,
        step_error=nmt.step_error,
        reason=None if nmt.converged else MAX_ITERATIONS,
        message=nmt.message,
        branch=_branch_of(theta) if nmt.converged else None,
        history=nmt.history,
    )
