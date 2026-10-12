"""Group C: joint-space trajectories (linear, cubic rest-to-rest).

Closed forms (SPEC section 3.5), with s = t/T and delta = qf - q0:
    linear: q = q0 + delta s,                  qd = delta/T,                 qdd = 0
    cubic:  q = q0 + delta (3 s^2 - 2 s^3),    qd = delta (6 s - 6 s^2)/T,   qdd = delta (6 - 12 s)/T^2
"""

import math

import numpy as np
import pytest

from manipulator_lab import Branch, PlanarArm, fk, trajectory
from manipulator_lab.kinematics import BEYOND_MAX_REACH, INSIDE_MIN_REACH

ARM = PlanarArm()
Q0 = np.array([0.3, -0.4])
QF = np.array([1.1, -1.5])
T = 2.0
END_TOL = 1e-12


def _grid(n=201, duration=T):
    return trajectory.time_grid(duration, n)


PROFILES = [trajectory.linear, trajectory.cubic]


# ------------------------------------------------------------------ grid
def test_time_grid_has_exact_endpoints():
    t = trajectory.time_grid(T, 11)
    assert t.shape == (11,)
    assert t[0] == 0.0 and t[-1] == T
    assert np.all(np.diff(t) > 0)


@pytest.mark.parametrize("n", [0, 1, -3, 2.5, True, "5"])
def test_time_grid_rejects_bad_count(n):
    with pytest.raises(ValueError):
        trajectory.time_grid(T, n)


# ------------------------------------------------------------- endpoints
@pytest.mark.parametrize("profile", PROFILES)
def test_endpoint_positions(profile):
    traj = profile(Q0, QF, T, _grid())
    assert np.max(np.abs(traj.q[0] - Q0)) <= END_TOL
    assert np.max(np.abs(traj.q[-1] - QF)) <= END_TOL


def test_cubic_is_rest_to_rest():
    traj = trajectory.cubic(Q0, QF, T, _grid())
    assert np.max(np.abs(traj.qd[0])) <= END_TOL
    assert np.max(np.abs(traj.qd[-1])) <= END_TOL


def test_linear_has_constant_velocity_and_zero_acceleration():
    traj = trajectory.linear(Q0, QF, T, _grid())
    expected = (QF - Q0) / T
    assert np.max(np.abs(traj.qd - expected)) <= END_TOL
    assert np.all(traj.qdd == 0.0)
    # velocity does not vanish at the ends: it jumps from rest (SPEC 3.5)
    assert np.all(np.abs(traj.qd[0]) > 0)


# --------------------------------------------------------- closed forms
@pytest.mark.parametrize("duration", [0.5, 2.0, 10.0])
def test_cubic_matches_closed_form(duration):
    t = _grid(duration=duration)
    traj = trajectory.cubic(Q0, QF, duration, t)
    s = (t / duration)[:, None]
    delta = QF - Q0
    assert np.max(np.abs(traj.q - (Q0 + delta * (3 * s**2 - 2 * s**3)))) <= 1e-12
    assert np.max(np.abs(traj.qd - delta * (6 * s - 6 * s**2) / duration)) <= 1e-12
    assert np.max(np.abs(traj.qdd - delta * (6 - 12 * s) / duration**2)) <= 1e-12


@pytest.mark.parametrize("duration", [0.5, 2.0, 10.0])
def test_closed_form_coefficients_match_the_nmt_4x4_system(duration):
    # Independent route: solve the 4x4 boundary-condition system with NMT's Gaussian elimination.
    for q0, qf in [(0.3, 1.1), (-0.4, -1.5), (0.0, 0.0), (2.0, -2.0)]:
        closed = trajectory.cubic_coefficients(q0, qf, duration)
        via_nmt = trajectory.cubic_coefficients_via_linear_system(q0, qf, duration)
        scale = max(1.0, float(np.max(np.abs(closed))))
        assert np.max(np.abs(closed - via_nmt)) <= 1e-9 * scale


def test_nmt_coefficients_reproduce_the_trajectory():
    t = _grid()
    traj = trajectory.cubic(Q0, QF, T, t)
    for j in range(2):
        a = trajectory.cubic_coefficients_via_linear_system(Q0[j], QF[j], T)
        q = a[0] + a[1] * t + a[2] * t**2 + a[3] * t**3
        qd = a[1] + 2 * a[2] * t + 3 * a[3] * t**2
        assert np.max(np.abs(q - traj.q[:, j])) <= 1e-12
        assert np.max(np.abs(qd - traj.qd[:, j])) <= 1e-12


# ------------------------------------------------------ internal consistency
@pytest.mark.parametrize("profile", PROFILES)
def test_velocity_is_the_derivative_of_position(profile):
    h = 1e-4
    t = np.linspace(0.1, T - 0.1, 50)
    plus = profile(Q0, QF, T, t + h)
    minus = profile(Q0, QF, T, t - h)
    mid = profile(Q0, QF, T, t)
    central = (plus.q - minus.q) / (2 * h)
    bound = h**2 * np.max(np.abs(12 * (QF - Q0) / T**3)) / 6 + 1e-9  # truncation + rounding
    assert np.max(np.abs(central - mid.qd)) <= bound


def test_acceleration_is_the_derivative_of_velocity_for_cubic():
    h = 1e-4
    t = np.linspace(0.1, T - 0.1, 50)
    central = (
        trajectory.cubic(Q0, QF, T, t + h).qd - trajectory.cubic(Q0, QF, T, t - h).qd
    ) / (2 * h)
    assert np.max(np.abs(central - trajectory.cubic(Q0, QF, T, t).qdd)) <= 1e-8


@pytest.mark.parametrize("profile", PROFILES)
def test_position_and_velocity_are_continuous_on_a_dense_grid(profile):
    t = _grid(n=2001)
    traj = profile(Q0, QF, T, t)
    dt = np.diff(t)[:, None]
    vmax = np.max(np.abs(traj.qd), axis=0)
    amax = 6.0 * np.abs(QF - Q0) / T**2
    assert np.all(np.abs(np.diff(traj.q, axis=0)) <= vmax * dt + 1e-12)
    if profile is trajectory.cubic:
        assert np.all(np.abs(np.diff(traj.qd, axis=0)) <= amax * dt + 1e-12)


# --------------------------------------------------------- cubic metrics
def test_cubic_peak_velocity_and_acceleration_match_theory():
    t = _grid(n=2001)
    traj = trajectory.cubic(Q0, QF, T, t)
    delta = np.abs(QF - Q0)
    assert np.max(np.abs(traj.qd), axis=0) == pytest.approx(1.5 * delta / T, rel=1e-12)
    assert np.max(np.abs(traj.qdd), axis=0) == pytest.approx(6.0 * delta / T**2, rel=1e-12)
    mid = trajectory.cubic(Q0, QF, T, np.array([T / 2]))
    assert np.max(np.abs(mid.q[0] - (Q0 + QF) / 2)) <= 1e-12


def test_cubic_is_monotone_without_overshoot():
    traj = trajectory.cubic(Q0, QF, T, _grid(n=1001))
    delta = QF - Q0
    assert np.all(np.diff(traj.q, axis=0) * np.sign(delta) >= -1e-15)
    low, high = np.minimum(Q0, QF), np.maximum(Q0, QF)
    assert np.all(traj.q >= low - 1e-12) and np.all(traj.q <= high + 1e-12)


def test_cubic_depends_only_on_normalised_time():
    a = trajectory.cubic(Q0, QF, 2.0, np.linspace(0, 2.0, 21))
    b = trajectory.cubic(Q0, QF, 8.0, np.linspace(0, 8.0, 21))
    assert np.max(np.abs(a.q - b.q)) <= 1e-12
    assert np.max(np.abs(a.qd - 4.0 * b.qd)) <= 1e-12  # velocity scales as 1/T


@pytest.mark.parametrize("profile", PROFILES)
def test_zero_displacement_stays_at_rest(profile):
    traj = profile(Q0, Q0, T, _grid())
    assert np.max(np.abs(traj.q - Q0)) <= END_TOL
    assert np.max(np.abs(traj.qd)) <= END_TOL
    assert np.max(np.abs(traj.qdd)) <= END_TOL


@pytest.mark.parametrize("profile", PROFILES)
def test_reversing_the_motion_mirrors_the_trajectory(profile):
    t = _grid()
    fwd = profile(Q0, QF, T, t)
    back = profile(QF, Q0, T, T - t)
    assert np.max(np.abs(fwd.q - back.q)) <= 1e-12


# ------------------------------------------------------------- result shape
@pytest.mark.parametrize("profile", PROFILES)
def test_result_arrays_are_aligned_and_finite(profile):
    t = _grid(n=7)
    traj = profile(Q0, QF, T, t)
    assert traj.t.shape == (7,)
    assert traj.q.shape == traj.qd.shape == traj.qdd.shape == (7, 2)
    assert all(np.all(np.isfinite(a)) for a in (traj.t, traj.q, traj.qd, traj.qdd))
    assert traj.profile in ("linear", "cubic")


def test_unordered_time_samples_are_evaluated_pointwise():
    t = np.array([1.5, 0.0, 2.0, 0.5])
    traj = trajectory.cubic(Q0, QF, T, t)
    ref = trajectory.cubic(Q0, QF, T, np.sort(t))
    assert np.max(np.abs(traj.q[np.argsort(t)] - ref.q)) <= 1e-15


# ------------------------------------------------------- input validation
@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("bad_T", [0.0, -1.0, math.nan, math.inf, True, "2"])
def test_invalid_duration_is_rejected(profile, bad_T):
    with pytest.raises(ValueError):
        profile(Q0, QF, bad_T, np.array([0.0]))


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("bad_t", [[-0.1], [T + 0.1], [math.nan], [0.0, math.inf], [], [[0.0, 1.0]]])
def test_invalid_time_samples_are_rejected(profile, bad_t):
    with pytest.raises(ValueError):
        profile(Q0, QF, T, bad_t)


@pytest.mark.parametrize("profile", PROFILES)
@pytest.mark.parametrize("bad_q", [(math.nan, 0.0), (0.0,), (0.0, 0.0, 0.0), "ab"])
def test_invalid_joint_vectors_are_rejected(profile, bad_q):
    with pytest.raises(ValueError):
        profile(bad_q, QF, T, np.array([0.0]))
    with pytest.raises(ValueError):
        profile(Q0, bad_q, T, np.array([0.0]))


def test_invalid_coefficient_inputs_are_rejected():
    for fn in (trajectory.cubic_coefficients, trajectory.cubic_coefficients_via_linear_system):
        with pytest.raises(ValueError):
            fn(0.0, 1.0, 0.0)
        with pytest.raises(ValueError):
            fn(math.nan, 1.0, 1.0)


# ------------------------------------------------- planning to a task target
@pytest.mark.parametrize("profile", ["linear", "cubic"])
def test_plan_to_target_reaches_the_target_on_the_elbow_up_branch(profile):
    target = (1.0, 0.5)
    plan = trajectory.plan_to_target(ARM, Q0, target, T, _grid(), profile=profile)
    assert plan.ok and plan.reason is None and plan.trajectory is not None
    assert plan.branch is Branch.ELBOW_UP and plan.theta_target[1] <= 0
    assert np.max(np.abs(plan.trajectory.q[0] - Q0)) <= END_TOL
    assert np.max(np.abs(plan.trajectory.q[-1] - plan.theta_target)) <= END_TOL
    assert fk(ARM, plan.trajectory.q[-1]) == pytest.approx(target, abs=1e-9)
    assert plan.trajectory.profile == profile
    if profile == "linear":
        assert np.max(np.abs(plan.trajectory.qd - (plan.theta_target - Q0) / T)) <= END_TOL
    else:
        assert np.max(np.abs(plan.trajectory.qd[[0, -1]])) <= END_TOL


def test_plan_to_target_honours_the_requested_branch():
    plan = trajectory.plan_to_target(ARM, Q0, (1.0, 0.5), T, _grid(), branch=Branch.ELBOW_DOWN)
    assert plan.ok and plan.theta_target[1] >= 0 and plan.branch is Branch.ELBOW_DOWN


@pytest.mark.parametrize("target, reason", [((2.0, 0.0), BEYOND_MAX_REACH), ((0.1, 0.0), INSIDE_MIN_REACH)])
def test_plan_to_unreachable_target_returns_a_reason_and_no_trajectory(target, reason):
    plan = trajectory.plan_to_target(ARM, Q0, target, T, _grid())
    assert not plan.ok and plan.reason == reason
    assert plan.trajectory is None and plan.theta_target is None


def test_plan_to_target_rejects_unknown_profile_and_bad_inputs():
    with pytest.raises(ValueError):
        trajectory.plan_to_target(ARM, Q0, (1.0, 0.5), T, _grid(), profile="quintic")
    with pytest.raises(ValueError):
        trajectory.plan_to_target(ARM, Q0, (math.nan, 0.5), T, _grid())
    with pytest.raises(ValueError):
        trajectory.plan_to_target(ARM, Q0, (1.0, 0.5), -1.0, _grid())
    with pytest.raises(ValueError):
        trajectory.plan_to_target(ARM, Q0, (1.0, 0.5), T, _grid(), branch="elbow_up")
