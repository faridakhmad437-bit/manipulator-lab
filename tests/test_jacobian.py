"""Week 2, part 1: Jacobian, singularity detection, conditioning.

cond(J) of the planar 2-link arm depends only on theta2. That is derived by hand
(SPEC section 3.4) and cross-checked here against an SVD-based implementation.
"""

import math

import numpy as np
import pytest

from manipulator_lab import (
    SINGULARITY_TOL,
    PlanarArm,
    condition_from_theta2,
    condition_number,
    fk,
    is_singular,
    jacobian,
)

ARM = PlanarArm()  # l1 = 1.0, l2 = 0.8


# ---------------------------------------------------------------- Jacobian
def test_jacobian_known_poses():
    # fully stretched along +x: only the y-row is non-zero
    assert jacobian(ARM, (0.0, 0.0)) == pytest.approx(np.array([[0.0, 0.0], [1.8, 0.8]]), abs=1e-12)
    # elbow bent 90 degrees: hand-computed
    assert jacobian(ARM, (0.0, math.pi / 2)) == pytest.approx(
        np.array([[-0.8, -0.8], [1.0, 0.0]]), abs=1e-12
    )


def test_jacobian_matches_central_finite_difference():
    rng = np.random.default_rng(0)
    h = 1e-6
    for theta in rng.uniform(-math.pi, math.pi, size=(50, 2)):
        numeric = np.empty((2, 2))
        for j in range(2):
            step = np.zeros(2)
            step[j] = h
            numeric[:, j] = (fk(ARM, theta + step) - fk(ARM, theta - step)) / (2 * h)
        assert jacobian(ARM, theta) == pytest.approx(numeric, abs=1e-8)


def test_determinant_is_l1_l2_sin_theta2():
    for t1 in np.linspace(-math.pi, math.pi, 13):
        for t2 in np.linspace(-math.pi, math.pi, 13):
            det = np.linalg.det(jacobian(ARM, (t1, t2)))
            assert det == pytest.approx(ARM.l1 * ARM.l2 * math.sin(t2), abs=1e-12)


@pytest.mark.parametrize("bad", [(0.0,), (math.nan, 0.0), (0.0, math.inf), "ab"])
def test_jacobian_rejects_invalid_theta(bad):
    with pytest.raises(ValueError):
        jacobian(ARM, bad)
    with pytest.raises(ValueError):
        condition_number(ARM, bad)
    with pytest.raises(ValueError):
        is_singular(bad)


# ----------------------------------------------------------- singularity flag
def test_singularity_tolerance_is_documented_value():
    assert SINGULARITY_TOL == 1e-6


@pytest.mark.parametrize("t2", [0.0, math.pi, -math.pi, 1e-9, math.pi - 1e-9])
def test_singular_poses_are_flagged(t2):
    assert is_singular((0.3, t2))


@pytest.mark.parametrize("t2", [0.01, -0.01, math.pi / 2, 3.0, -2.0])
def test_regular_poses_are_not_flagged(t2):
    assert not is_singular((0.3, t2))


def test_singularity_threshold_is_adjustable():
    assert is_singular((0.0, 0.05), tol=0.1)
    assert not is_singular((0.0, 0.05), tol=0.01)


# -------------------------------------------------------------- conditioning
def test_condition_is_at_least_one():
    for t2 in np.linspace(-3.0, 3.0, 41):
        assert condition_number(ARM, (0.2, t2)) >= 1.0


def test_condition_is_independent_of_theta1():
    t2 = 0.8
    values = [condition_number(ARM, (t1, t2)) for t1 in np.linspace(-math.pi, math.pi, 17)]
    assert values == pytest.approx([values[0]] * len(values), rel=1e-9)


def test_condition_is_symmetric_in_theta2_sign():
    for t2 in (0.3, 1.0, 2.5):
        assert condition_number(ARM, (0.0, t2)) == pytest.approx(
            condition_number(ARM, (0.0, -t2)), rel=1e-9
        )


def test_condition_grows_toward_singularity():
    values = [condition_number(ARM, (0.0, t2)) for t2 in (1.0, 0.1, 0.01, 1e-3)]
    assert values == sorted(values)
    assert len(set(values)) == len(values)


@pytest.mark.parametrize("t1", [0.0, 0.7, -2.0])
@pytest.mark.parametrize("t2", [0.0, math.pi, -math.pi])
def test_condition_is_infinite_when_numerically_rank_deficient(t1, t2):
    assert math.isinf(condition_number(ARM, (t1, t2)))
    assert math.isinf(condition_from_theta2(ARM, t2))


def test_condition_is_finite_just_outside_singularity():
    value = condition_number(ARM, (0.0, 1e-6))
    assert math.isfinite(value) and value > 1e5


# ------------------------------------------- closed form vs SVD cross-check
@pytest.mark.parametrize("arm", [PlanarArm(), PlanarArm(1.0, 1.0), PlanarArm(0.5, 2.0)])
def test_closed_form_matches_svd_away_from_singularity(arm):
    for t2 in np.linspace(-math.pi, math.pi, 61):
        if abs(math.sin(t2)) < 1e-3:
            continue
        assert condition_from_theta2(arm, t2) == pytest.approx(
            condition_number(arm, (0.4, t2)), rel=1e-9
        )


def test_closed_form_rejects_invalid_theta2():
    for bad in (math.nan, math.inf, "x"):
        with pytest.raises(ValueError):
            condition_from_theta2(ARM, bad)
