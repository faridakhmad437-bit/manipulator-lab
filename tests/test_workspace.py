"""Week 2, part 2: workspace boundary and conditioning map."""

import math

import numpy as np
import pytest

from manipulator_lab import (
    Branch,
    PlanarArm,
    condition_from_theta2,
    condition_map,
    is_reachable,
    workspace_boundary,
)

ARM = PlanarArm()  # reach 0.2 <= r <= 1.8


# ----------------------------------------------------------------- boundary
def test_boundary_circles_have_correct_radii_and_are_closed():
    outer, inner = workspace_boundary(ARM, n=181)
    assert outer.shape == inner.shape == (181, 2)
    assert np.hypot(*outer.T) == pytest.approx(ARM.max_reach, abs=1e-12)
    assert np.hypot(*inner.T) == pytest.approx(ARM.min_reach, abs=1e-12)
    assert outer[0] == pytest.approx(outer[-1], abs=1e-12)
    assert inner[0] == pytest.approx(inner[-1], abs=1e-12)


def test_boundary_of_equal_links_has_degenerate_inner_circle():
    _, inner = workspace_boundary(PlanarArm(1.0, 1.0), n=11)
    assert np.all(inner == 0.0)


@pytest.mark.parametrize("bad", [2, 1, 0, -5, 3.5, True, "10"])
def test_boundary_rejects_invalid_point_count(bad):
    with pytest.raises(ValueError):
        workspace_boundary(ARM, n=bad)


# -------------------------------------------------------------------- map
def test_map_shapes_and_grid_symmetry():
    cmap = condition_map(ARM, resolution=41)
    assert cmap.cond.shape == cmap.reachable.shape == (41, 41)
    assert cmap.x.shape == cmap.y.shape == (41,)
    assert cmap.x == pytest.approx(-cmap.x[::-1], abs=1e-12)
    assert cmap.x[-1] > ARM.max_reach  # default extent covers the whole workspace


def test_map_reachable_mask_matches_is_reachable():
    cmap = condition_map(ARM, resolution=41)
    for iy, y in enumerate(cmap.y):
        for ix, x in enumerate(cmap.x):
            assert bool(cmap.reachable[iy, ix]) == is_reachable(ARM, (x, y))


def test_map_is_nan_exactly_where_unreachable():
    cmap = condition_map(ARM, resolution=41)
    assert np.array_equal(np.isnan(cmap.cond), ~cmap.reachable)
    assert np.all(cmap.cond[cmap.reachable] >= 1.0)
    assert cmap.reachable.any() and (~cmap.reachable).any()


def test_map_is_radially_symmetric():
    cond = condition_map(ARM, resolution=41).cond
    np.testing.assert_allclose(cond, cond.T, rtol=1e-9, equal_nan=True)
    np.testing.assert_allclose(cond, cond[::-1, :], rtol=1e-9, equal_nan=True)
    np.testing.assert_allclose(cond, cond[:, ::-1], rtol=1e-9, equal_nan=True)


def test_map_is_identical_for_both_branches():
    up = condition_map(ARM, resolution=31, branch=Branch.ELBOW_UP).cond
    down = condition_map(ARM, resolution=31, branch=Branch.ELBOW_DOWN).cond
    np.testing.assert_allclose(up, down, rtol=1e-9, equal_nan=True)


def _theta2_at_radius(arm, r):
    """|theta2| of the IK solution at radius r (law of cosines)."""
    c2 = (r * r - arm.l1**2 - arm.l2**2) / (2.0 * arm.l1 * arm.l2)
    return math.acos(c2)


def test_map_condition_rises_toward_both_workspace_boundaries():
    """Inside a band of width ``band`` next to a boundary, cond(J) must not fall below
    its closed-form value at the band edge (cond is monotone toward each boundary)."""
    band = 0.02
    cmap = condition_map(ARM, resolution=201)
    r = np.hypot(*np.meshgrid(cmap.x, cmap.y))
    inside = cmap.reachable
    mid = inside & (np.abs(r - 1.0) < 0.05)
    near_outer = inside & (r > ARM.max_reach - band)
    near_inner = inside & (r < ARM.min_reach + band)
    assert near_outer.any() and near_inner.any() and mid.any()

    edge_outer = condition_from_theta2(ARM, _theta2_at_radius(ARM, ARM.max_reach - band))
    edge_inner = condition_from_theta2(ARM, _theta2_at_radius(ARM, ARM.min_reach + band))
    assert np.nanmin(cmap.cond[near_outer]) >= edge_outer * (1 - 1e-9)
    assert np.nanmin(cmap.cond[near_inner]) >= edge_inner * (1 - 1e-9)

    mid_median = np.nanmedian(cmap.cond[mid])
    assert np.nanmedian(cmap.cond[near_outer]) > 5 * mid_median
    assert np.nanmedian(cmap.cond[near_inner]) > 5 * mid_median


def test_equal_links_origin_cell_is_reachable_and_singular():
    cmap = condition_map(PlanarArm(1.0, 1.0), resolution=41)
    assert cmap.reachable[20, 20]
    assert math.isinf(cmap.cond[20, 20])
    assert not np.any(np.isnan(cmap.cond[cmap.reachable]))


def test_map_custom_extent():
    cmap = condition_map(ARM, resolution=11, extent=1.0)
    assert cmap.x[0] == pytest.approx(-1.0) and cmap.x[-1] == pytest.approx(1.0)


@pytest.mark.parametrize("bad", [1, 0, -3, 2.5, True, "10", None])
def test_map_rejects_invalid_resolution(bad):
    with pytest.raises(ValueError):
        condition_map(ARM, resolution=bad)


@pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf, "2"])
def test_map_rejects_invalid_extent(bad):
    with pytest.raises(ValueError):
        condition_map(ARM, resolution=11, extent=bad)
