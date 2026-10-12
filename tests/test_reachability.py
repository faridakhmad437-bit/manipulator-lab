"""Group B: reachability, boundaries, degenerate arms, input validation."""

import math

import numpy as np
import pytest

from manipulator_lab import (
    REACH_TOL,
    Branch,
    PlanarArm,
    fk,
    ik_analytic,
    is_reachable,
)

ARM = PlanarArm()  # l1 = 1.0, l2 = 0.8 -> 0.2 <= r <= 1.8
POS_TOL = 1e-9


def _polar(r, angle):
    return (r * math.cos(angle), r * math.sin(angle))


# ------------------------------------------------------------ arm geometry
def test_default_arm_matches_spec():
    assert (ARM.l1, ARM.l2) == (1.0, 0.8)
    assert ARM.max_reach == pytest.approx(1.8)
    assert ARM.min_reach == pytest.approx(0.2)


def test_reach_tolerance_is_tight():
    assert REACH_TOL == 1e-9


# ------------------------------------------------------------ inside / outside
def test_point_inside_workspace_is_reachable():
    assert is_reachable(ARM, (1.0, 0.5))
    result = ik_analytic(ARM, (1.0, 0.5))
    assert result.reachable and result.reason is None
    assert fk(ARM, result.theta) == pytest.approx((1.0, 0.5), abs=POS_TOL)


@pytest.mark.parametrize("angle", np.linspace(-math.pi, math.pi, 13))
def test_outer_boundary_is_reachable_at_all_angles(angle):
    target = _polar(ARM.max_reach, angle)
    assert is_reachable(ARM, target)
    result = ik_analytic(ARM, target)
    assert result.reachable
    assert result.theta[1] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize("angle", np.linspace(-math.pi, math.pi, 13))
def test_inner_boundary_is_reachable_at_all_angles(angle):
    target = _polar(ARM.min_reach, angle)
    assert is_reachable(ARM, target)
    result = ik_analytic(ARM, target)
    assert result.reachable
    assert abs(result.theta[1]) == pytest.approx(math.pi, abs=1e-6)
    assert fk(ARM, result.theta) == pytest.approx(target, abs=POS_TOL)


def test_slightly_beyond_outer_boundary_is_rejected_with_reason():
    result = ik_analytic(ARM, (ARM.max_reach + 1e-6, 0.0))
    assert not is_reachable(ARM, (ARM.max_reach + 1e-6, 0.0))
    assert not result.reachable
    assert result.theta is None
    assert result.reason == "target_beyond_max_reach"


def test_slightly_inside_inner_boundary_is_rejected_with_reason():
    result = ik_analytic(ARM, (ARM.min_reach - 1e-6, 0.0))
    assert not result.reachable
    assert result.theta is None
    assert result.reason == "target_inside_min_reach"


def test_origin_is_unreachable_when_links_differ():
    result = ik_analytic(ARM, (0.0, 0.0))
    assert not result.reachable
    assert result.reason == "target_inside_min_reach"


def test_far_target_is_unreachable():
    assert not is_reachable(ARM, (5.0, 5.0))


def test_boundary_tolerance_absorbs_float_noise_only():
    assert is_reachable(ARM, (ARM.max_reach + 1e-12, 0.0))
    assert not is_reachable(ARM, (ARM.max_reach + 1e-8, 0.0))


# ---------------------------------------------------------------- consistency
def test_is_reachable_agrees_with_ik_and_geometry():
    rng = np.random.default_rng(0)
    points = rng.uniform(-2.0, 2.0, size=(2000, 2))
    for p in points:
        r = math.hypot(*p)
        if min(abs(r - ARM.min_reach), abs(r - ARM.max_reach)) < 1e-6:
            continue  # boundary is covered by dedicated tests
        expected = ARM.min_reach <= r <= ARM.max_reach
        assert is_reachable(ARM, p) is expected
        assert ik_analytic(ARM, p).reachable is expected
        assert ik_analytic(ARM, p, Branch.ELBOW_DOWN).reachable is expected


# ------------------------------------------------------- l1 == l2 (degenerate)
def test_equal_links_origin_is_reachable_without_nan():
    arm = PlanarArm(1.0, 1.0)
    assert arm.min_reach == 0.0
    for branch in (Branch.ELBOW_UP, Branch.ELBOW_DOWN):
        result = ik_analytic(arm, (0.0, 0.0), branch)
        assert result.reachable
        assert np.all(np.isfinite(result.theta))
        assert fk(arm, result.theta) == pytest.approx((0.0, 0.0), abs=POS_TOL)


def test_equal_links_outer_boundary_and_beyond():
    arm = PlanarArm(1.0, 1.0)
    assert is_reachable(arm, (2.0, 0.0))
    assert not is_reachable(arm, (2.1, 0.0))


# --------------------------------------------------------- invalid target input
@pytest.mark.parametrize(
    "bad",
    [(math.nan, 0.0), (0.0, math.inf), (1.0,), (1.0, 2.0, 3.0), ((1.0, 2.0), (3.0, 4.0)), "xy"],
)
def test_invalid_target_is_rejected(bad):
    with pytest.raises(ValueError):
        ik_analytic(ARM, bad)
    with pytest.raises(ValueError):
        is_reachable(ARM, bad)


# ----------------------------------------------------------- invalid arm input
@pytest.mark.parametrize(
    "l1, l2",
    [
        (0.0, 1.0),
        (1.0, -1.0),
        (math.nan, 1.0),
        (1.0, math.inf),
        (True, 1.0),
        ("1", 1.0),
        (None, 1.0),
    ],
)
def test_invalid_link_lengths_are_rejected(l1, l2):
    with pytest.raises(ValueError):
        PlanarArm(l1, l2)


def test_arm_is_immutable():
    with pytest.raises(AttributeError):
        ARM.l1 = 2.0  # type: ignore[misc]
