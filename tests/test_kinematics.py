"""Group A: FK <-> IK consistency.

Round trips are compared in task space (end-effector position), not joint space:
near the workspace boundary the joint angles are ill-conditioned (arccos-type
sensitivity) while the position stays accurate.
"""

import math

import numpy as np
import pytest

from manipulator_lab import Branch, PlanarArm, fk, ik_analytic

POS_TOL = 1e-9  # metres, see SPEC section 6

ARM = PlanarArm()  # l1 = 1.0, l2 = 0.8


# ---------------------------------------------------------------- FK values
@pytest.mark.parametrize(
    "theta, expected",
    [
        ((0.0, 0.0), (1.8, 0.0)),  # fully stretched along +x
        ((math.pi / 2, 0.0), (0.0, 1.8)),  # fully stretched along +y
        ((0.0, math.pi / 2), (1.0, 0.8)),  # elbow bent 90 degrees CCW
        ((0.0, math.pi), (0.2, 0.0)),  # folded back
        ((0.0, -math.pi / 2), (1.0, -0.8)),  # elbow bent 90 degrees CW
    ],
)
def test_fk_known_values(theta, expected):
    assert fk(ARM, theta) == pytest.approx(expected, abs=POS_TOL)


def test_fk_is_periodic_in_joint_angles():
    theta = np.array([0.7, -1.3])
    shifted = theta + 2 * math.pi
    assert fk(ARM, shifted) == pytest.approx(fk(ARM, theta), abs=POS_TOL)


@pytest.mark.parametrize(
    "bad",
    [(0.0,), (0.0, 0.0, 0.0), ((0.0, 0.0), (0.0, 0.0)), (math.nan, 0.0), (0.0, math.inf), "ab"],
)
def test_fk_rejects_invalid_theta(bad):
    with pytest.raises(ValueError):
        fk(ARM, bad)


# ------------------------------------------------------- FK -> IK -> FK grid
_ANGLES = np.linspace(-math.pi, math.pi, 25)  # includes 0, +-pi (singular poses)


@pytest.mark.parametrize("branch", [Branch.ELBOW_UP, Branch.ELBOW_DOWN])
def test_roundtrip_on_joint_grid_in_task_space(branch):
    for t1 in _ANGLES:
        for t2 in _ANGLES:
            target = fk(ARM, (t1, t2))
            result = ik_analytic(ARM, target, branch)
            assert result.reachable, (t1, t2, result.reason)
            assert fk(ARM, result.theta) == pytest.approx(target, abs=POS_TOL)


def test_both_branches_reach_the_same_point():
    target = fk(ARM, (0.4, -0.8))
    up = ik_analytic(ARM, target, Branch.ELBOW_UP)
    down = ik_analytic(ARM, target, Branch.ELBOW_DOWN)
    assert fk(ARM, up.theta) == pytest.approx(target, abs=POS_TOL)
    assert fk(ARM, down.theta) == pytest.approx(target, abs=POS_TOL)
    assert up.theta[1] < 0 < down.theta[1]


# ------------------------------------------------------ branch convention
def test_elbow_up_is_negative_theta2():
    target = fk(ARM, (0.6, -0.9))
    result = ik_analytic(ARM, target, Branch.ELBOW_UP)
    assert result.theta == pytest.approx((0.6, -0.9), abs=1e-9)


def test_elbow_down_is_positive_theta2():
    target = fk(ARM, (0.6, -0.9))
    result = ik_analytic(ARM, target, Branch.ELBOW_DOWN)
    assert result.theta[1] == pytest.approx(0.9, abs=1e-9)
    assert fk(ARM, result.theta) == pytest.approx(target, abs=POS_TOL)


def test_default_branch_is_elbow_up():
    target = fk(ARM, (0.6, -0.9))
    assert ik_analytic(ARM, target).branch is Branch.ELBOW_UP


# ------------------------------------------------------ singular poses
def test_fully_stretched_pose_has_identical_branches():
    target = fk(ARM, (0.3, 0.0))
    up = ik_analytic(ARM, target, Branch.ELBOW_UP)
    down = ik_analytic(ARM, target, Branch.ELBOW_DOWN)
    assert up.theta == pytest.approx((0.3, 0.0), abs=1e-7)
    assert down.theta == pytest.approx(up.theta, abs=1e-7)


def test_folded_pose_is_reachable_with_theta2_pi():
    target = fk(ARM, (0.3, math.pi))
    for branch in (Branch.ELBOW_UP, Branch.ELBOW_DOWN):
        result = ik_analytic(ARM, target, branch)
        assert result.reachable
        assert abs(result.theta[1]) == pytest.approx(math.pi, abs=1e-7)
        assert fk(ARM, result.theta) == pytest.approx(target, abs=POS_TOL)


# ------------------------------------------------------ angle wrap-around
@pytest.mark.parametrize("t1", [math.pi - 1e-3, -math.pi + 1e-3, math.pi, -math.pi])
def test_wraparound_near_plus_minus_pi(t1):
    target = fk(ARM, (t1, -0.5))
    for branch in (Branch.ELBOW_UP, Branch.ELBOW_DOWN):
        result = ik_analytic(ARM, target, branch)
        assert result.reachable
        assert -math.pi <= result.theta[0] <= math.pi
        assert -math.pi <= result.theta[1] <= math.pi
        assert fk(ARM, result.theta) == pytest.approx(target, abs=POS_TOL)
