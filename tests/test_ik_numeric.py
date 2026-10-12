"""Group D: numerical IK (Newton from the pinned NMT release) against the analytic solution.

Facts established by reading the NMT v1.0.0 source and by smoke tests (SPEC section 4):
the branch Newton lands on is decided by the initial guess; the NMT default initial guess
(0, 0) is exactly singular and raises; unreachable targets are not attempted.
"""

import math

import numerical_toolkit
import numpy as np
import pytest
from numerical_toolkit import SolverOptions, planar_2link_ik

from manipulator_lab import (
    BEYOND_MAX_REACH,
    INSIDE_MIN_REACH,
    MAX_ITERATIONS,
    NON_FINITE_VALUE,
    SINGULAR_JACOBIAN,
    Branch,
    NumericIKResult,
    PlanarArm,
    default_initial_guess,
    fk,
    ik_analytic,
    ik_newton,
    is_singular,
    wrap_to_pi,
)

ARM = PlanarArm()  # l1 = 1.0, l2 = 0.8
POS_TOL = 1e-9
ANGLE_TOL = 1e-6
FAILURE_REASONS = {MAX_ITERATIONS, SINGULAR_JACOBIAN, NON_FINITE_VALUE}


def _polar(r, angle):
    return (r * math.cos(angle), r * math.sin(angle))


def _wrapped_diff(a, b):
    return max(abs(wrap_to_pi(float(x) - float(y))) for x, y in zip(a, b))


def _matches_analytic(arm, target, theta, tol=ANGLE_TOL):
    return any(
        _wrapped_diff(theta, ik_analytic(arm, target, branch).theta) <= tol for branch in Branch
    )


# --------------------------------------------------------------- dependency
def test_nmt_dependency_is_the_pinned_release():
    assert numerical_toolkit.__version__ == "1.0.0"


# ------------------------------------------------------- branch selection
def test_initial_guess_decides_the_branch():
    target = fk(ARM, (0.6, -0.9))
    up = ik_newton(ARM, target, initial=(0.3, -0.5))
    down = ik_newton(ARM, target, initial=(0.3, 0.5))
    assert up.converged and down.converged
    assert up.branch is Branch.ELBOW_UP
    assert down.branch is Branch.ELBOW_DOWN
    assert up.theta == pytest.approx((0.6, -0.9), abs=1e-8)
    assert down.theta == pytest.approx(ik_analytic(ARM, target, Branch.ELBOW_DOWN).theta, abs=1e-8)


def test_default_initial_guess_is_regular_and_follows_the_branch():
    up = default_initial_guess(ARM, (1.0, 0.5), Branch.ELBOW_UP)
    down = default_initial_guess(ARM, (1.0, 0.5), Branch.ELBOW_DOWN)
    assert up.shape == down.shape == (2,)
    assert np.all(np.isfinite(up)) and np.all(np.isfinite(down))
    assert up[1] < 0 < down[1]
    assert not is_singular(up) and not is_singular(down)


@pytest.mark.parametrize("branch", list(Branch))
def test_default_guess_reaches_requested_branch_in_mid_annulus(branch):
    for radius in np.linspace(0.5, 1.5, 6):
        for angle in np.linspace(-math.pi, math.pi, 13)[:-1]:
            target = _polar(radius, angle)
            result = ik_newton(ARM, target, branch=branch)
            assert result.converged, (radius, angle, result.reason)
            assert result.branch is branch
            expected = ik_analytic(ARM, target, branch).theta
            assert _wrapped_diff(result.theta, expected) <= ANGLE_TOL


# ----------------------------------------------------- converged result
def test_converged_result_is_self_consistent():
    target = fk(ARM, (0.6, -0.9))
    result = ik_newton(ARM, target, initial=(0.3, -0.5))
    assert isinstance(result, NumericIKResult)
    assert result.converged and result.reason is None
    assert result.residual is not None and result.residual <= POS_TOL
    assert fk(ARM, result.theta) == pytest.approx(target, abs=POS_TOL)
    assert -math.pi <= result.theta[0] <= math.pi and -math.pi <= result.theta[1] <= math.pi
    assert 0 < result.iterations <= 12
    assert result.history is not None and len(result.history) == result.iterations + 1


def test_adapter_agrees_with_nmt_planar_2link_ik():
    target = fk(ARM, (0.6, -0.9))
    initial = (0.3, -0.5)
    ours = ik_newton(ARM, target, initial=initial)
    nmt = planar_2link_ik(target, ARM.l1, ARM.l2, initial)
    assert nmt.converged and ours.converged
    assert ours.iterations == nmt.iterations
    assert _wrapped_diff(ours.theta, nmt.solution) <= 1e-9


def test_solution_is_wrapped_even_from_a_far_initial_guess():
    target = fk(ARM, (0.6, -0.9))
    result = ik_newton(ARM, target, initial=(0.3 + 4 * math.pi, -0.5 - 2 * math.pi))
    assert result.converged
    assert -math.pi <= result.theta[0] <= math.pi and -math.pi <= result.theta[1] <= math.pi
    assert result.theta == pytest.approx((0.6, -0.9), abs=1e-8)


# ------------------------------------------------------ failure reporting
@pytest.mark.parametrize(
    "target, reason",
    [((2.0, 0.0), BEYOND_MAX_REACH), ((0.1, 0.0), INSIDE_MIN_REACH), ((0.0, 0.0), INSIDE_MIN_REACH)],
)
def test_unreachable_targets_are_reported_not_attempted(target, reason):
    result = ik_newton(ARM, target)
    assert not result.converged
    assert result.reason == reason
    assert result.theta is None and result.residual is None and result.branch is None
    assert result.iterations == 0


def test_exactly_singular_initial_guess_becomes_a_structured_result():
    target = fk(ARM, (0.6, -0.9))
    result = ik_newton(ARM, target, initial=(0.0, 0.0))  # NMT raises on this input
    assert not result.converged
    assert result.reason == SINGULAR_JACOBIAN
    assert result.theta is None and result.iterations is None
    assert "singular" in result.message.lower()


def test_iteration_limit_is_reported_with_the_last_iterate():
    target = fk(ARM, (0.6, -0.9))
    result = ik_newton(ARM, target, initial=(0.3, -0.5), options=SolverOptions(maxiter=2))
    assert not result.converged
    assert result.reason == MAX_ITERATIONS
    assert result.iterations == 2
    assert result.theta is not None and np.all(np.isfinite(result.theta))
    assert result.branch is None


# ------------------------------------------- invariants over random starts
def test_random_starts_never_raise_and_converged_results_are_valid():
    rng = np.random.default_rng(0)
    converged = 0
    for _ in range(500):
        target = _polar(rng.uniform(0.25, 1.75), rng.uniform(-math.pi, math.pi))
        initial = rng.uniform(-math.pi, math.pi, size=2)
        result = ik_newton(ARM, target, initial=initial)
        if result.converged:
            converged += 1
            assert result.reason is None
            assert fk(ARM, result.theta) == pytest.approx(target, abs=1e-8)
            assert _matches_analytic(ARM, target, result.theta, tol=1e-5)
        else:
            assert result.reason in FAILURE_REASONS
    assert converged > 0  # the invariant above must have been exercised


@pytest.mark.parametrize("gap", [1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 0.0])
@pytest.mark.parametrize("boundary", ["outer", "inner"])
def test_targets_near_a_workspace_boundary_never_raise(gap, boundary):
    radius = ARM.max_reach - gap if boundary == "outer" else ARM.min_reach + gap
    result = ik_newton(ARM, (radius, 0.0))
    if result.converged:
        assert fk(ARM, result.theta) == pytest.approx((radius, 0.0), abs=1e-8)
    else:
        assert result.reason in FAILURE_REASONS


# ------------------------------------------------------ input validation
@pytest.mark.parametrize("bad", [(math.nan, 0.0), (0.0, math.inf), (1.0,), (1.0, 2.0, 3.0), "xy"])
def test_invalid_target_is_rejected(bad):
    with pytest.raises(ValueError):
        ik_newton(ARM, bad)


@pytest.mark.parametrize("bad", [(math.nan, 0.0), (0.0,), (0.0, 0.0, 0.0), "ab"])
def test_invalid_initial_guess_is_rejected(bad):
    with pytest.raises(ValueError):
        ik_newton(ARM, (1.0, 0.5), initial=bad)


@pytest.mark.parametrize("bad", ["options", {"maxiter": 5}, 3])
def test_invalid_options_are_rejected(bad):
    with pytest.raises(ValueError):
        ik_newton(ARM, (1.0, 0.5), options=bad)


def test_invalid_branch_is_rejected():
    with pytest.raises(ValueError):
        ik_newton(ARM, (1.0, 0.5), branch="elbow_up")
    with pytest.raises(ValueError):
        default_initial_guess(ARM, (1.0, 0.5), "elbow_up")
