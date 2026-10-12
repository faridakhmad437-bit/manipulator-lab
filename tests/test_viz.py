"""Visualisation helpers (SPEC section 9). Numerical parts are tested without drawing;
drawing is tested through the data the artists hold and through the files written."""

import math

import numpy as np
import pytest

pytest.importorskip("matplotlib")
Image = pytest.importorskip("PIL.Image")

from manipulator_lab import PlanarArm, fk, trajectory  # noqa: E402
from manipulator_lab import viz  # noqa: E402

ARM = PlanarArm()
Q0 = np.array([-0.5, -0.3])
TARGET = (-0.6, 1.0)
T = 2.0
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _plan(profile="cubic", target=TARGET):
    t = trajectory.time_grid(T, 201)
    plan = trajectory.plan_to_target(ARM, Q0, target, T, t, profile=profile)
    assert plan.ok
    return plan


# ------------------------------------------------------------- pure geometry
def test_arm_points_are_base_elbow_tip():
    rng = np.random.default_rng(0)
    for _ in range(50):
        q = rng.uniform(-math.pi, math.pi, 2)
        pts = viz.arm_points(ARM, q)
        assert pts.shape == (3, 2)
        assert np.all(pts[0] == 0.0)
        assert np.hypot(*pts[1]) == pytest.approx(ARM.l1, abs=1e-12)
        assert np.hypot(*(pts[2] - pts[1])) == pytest.approx(ARM.l2, abs=1e-12)
        assert pts[2] == pytest.approx(fk(ARM, q), abs=1e-12)


@pytest.mark.parametrize("bad", [(math.nan, 0.0), (0.0,), "ab"])
def test_arm_points_rejects_bad_angles(bad):
    with pytest.raises(ValueError):
        viz.arm_points(ARM, bad)


def test_frame_positions_follow_the_trajectory():
    traj = _plan().trajectory
    idx = np.array([0, 10, 100, 200])
    pos = viz.frame_positions(ARM, traj, idx)
    assert pos.shape == (4, 3, 2)
    for k, i in enumerate(idx):
        assert pos[k] == pytest.approx(viz.arm_points(ARM, traj.q[i]), abs=1e-15)


def test_frame_positions_default_is_every_sample():
    traj = _plan().trajectory
    assert viz.frame_positions(ARM, traj).shape == (201, 3, 2)


def test_frame_indices_cover_both_ends_in_order():
    idx = viz.frame_indices(201, 60)
    assert len(idx) == 60 and idx[0] == 0 and idx[-1] == 200
    assert np.all(np.diff(idx) > 0)
    assert np.array_equal(viz.frame_indices(5, 5), np.arange(5))


@pytest.mark.parametrize("n_samples, n_frames", [(10, 1), (10, 11), (1, 2), (10, 2.5), (10, True), (10, 0)])
def test_frame_indices_rejects_bad_counts(n_samples, n_frames):
    with pytest.raises(ValueError):
        viz.frame_indices(n_samples, n_frames)


# ----------------------------------------------------------------- animation
def test_animation_frames_match_kinematics():
    plan = _plan()
    anim = viz.animate_plan(ARM, plan, TARGET, n_frames=40)
    assert anim.positions.shape == (40, 3, 2)
    anim.set_frame(0)
    assert np.column_stack(anim.arm_line.get_data()) == pytest.approx(viz.arm_points(ARM, Q0), abs=1e-12)
    anim.set_frame(39)
    shown = np.column_stack(anim.arm_line.get_data())
    assert np.hypot(*(shown[2] - np.array(TARGET))) <= 1e-9  # last frame sits on the target


def test_every_rendered_frame_is_a_rigid_arm():
    anim = viz.animate_plan(ARM, _plan(), TARGET, n_frames=60)
    for i in range(60):
        anim.set_frame(i)
        pts = np.column_stack(anim.arm_line.get_data())
        assert np.hypot(*pts[1]) == pytest.approx(ARM.l1, abs=1e-12)
        assert np.hypot(*(pts[2] - pts[1])) == pytest.approx(ARM.l2, abs=1e-12)


def test_trail_grows_with_the_frame_index():
    anim = viz.animate_plan(ARM, _plan(), TARGET, n_frames=30)
    anim.set_frame(0)
    assert len(anim.trail_line.get_xdata()) == 1
    anim.set_frame(29)
    assert len(anim.trail_line.get_xdata()) == 30


def test_set_frame_rejects_out_of_range_index():
    anim = viz.animate_plan(ARM, _plan(), TARGET, n_frames=10)
    for bad in (-1, 10, 2.5, True):
        with pytest.raises(ValueError):
            anim.set_frame(bad)


def test_animate_plan_rejects_failed_or_invalid_plans():
    failed = trajectory.plan_to_target(ARM, Q0, (2.0, 0.0), T, trajectory.time_grid(T, 5))
    assert not failed.ok
    with pytest.raises(ValueError):
        viz.animate_plan(ARM, failed, (2.0, 0.0))
    with pytest.raises(ValueError):
        viz.animate_plan(ARM, "plan", TARGET)
    with pytest.raises(ValueError):
        viz.animate_plan(ARM, _plan(), (math.nan, 0.0))
    with pytest.raises(ValueError):
        viz.animate_plan(ARM, _plan(), TARGET, n_frames=1)


def test_axes_contain_the_whole_workspace():
    anim = viz.animate_plan(ARM, _plan(), TARGET, n_frames=10)
    ax = anim.fig.axes[0]
    for lim in (ax.get_xlim(), ax.get_ylim()):
        assert lim[0] <= -ARM.max_reach and lim[1] >= ARM.max_reach


# ---------------------------------------------------------------- file output
def test_save_gif_writes_a_valid_multi_frame_gif(tmp_path):
    anim = viz.animate_plan(ARM, _plan(), TARGET, n_frames=24)
    path = tmp_path / "demo.gif"
    viz.save_gif(anim, path, fps=12)
    assert path.stat().st_size > 0
    with Image.open(path) as gif:
        assert gif.format == "GIF"
        # identical consecutive frames may be merged by the encoder, so only bounds are exact
        assert 2 <= gif.n_frames <= 24


def _gif_duration_ms(path):
    with Image.open(path) as gif:
        total = 0
        for i in range(gif.n_frames):
            gif.seek(i)
            total += gif.info.get("duration", 0)
        return total


def test_hold_lengthens_the_pause_on_the_last_pose(tmp_path):
    anim = viz.animate_plan(ARM, _plan(), TARGET, n_frames=12)
    viz.save_gif(anim, tmp_path / "plain.gif", fps=10)
    viz.save_gif(anim, tmp_path / "held.gif", fps=10, hold=10)
    assert _gif_duration_ms(tmp_path / "held.gif") > _gif_duration_ms(tmp_path / "plain.gif")


def test_save_gif_rejects_bad_arguments(tmp_path):
    anim = viz.animate_plan(ARM, _plan(), TARGET, n_frames=5)
    with pytest.raises(ValueError):
        viz.save_gif(anim, tmp_path / "demo.png")
    for fps in (0, -3, math.nan, True):
        with pytest.raises(ValueError):
            viz.save_gif(anim, tmp_path / "demo.gif", fps=fps)
    for hold in (-1, 1.5, True):
        with pytest.raises(ValueError):
            viz.save_gif(anim, tmp_path / "demo.gif", hold=hold)


def test_snapshots_png_is_written(tmp_path):
    path = tmp_path / "snap.png"
    viz.plot_snapshots(ARM, _plan(), TARGET, path, n=5)
    assert path.read_bytes()[:8] == PNG_SIGNATURE
    with pytest.raises(ValueError):
        viz.plot_snapshots(ARM, _plan(), TARGET, path, n=1)


@pytest.mark.parametrize("target", [(1.6, 1.2), (0.05, 0.0)])
def test_unreachable_figure_reports_the_reason(tmp_path, target):
    plan = trajectory.plan_to_target(ARM, Q0, target, T, trajectory.time_grid(T, 5))
    assert not plan.ok
    path = tmp_path / "unreachable.png"
    fig = viz.plot_unreachable(ARM, Q0, plan, target, path)
    assert path.read_bytes()[:8] == PNG_SIGNATURE
    texts = " ".join(t.get_text() for t in fig.axes[0].texts) + fig.axes[0].get_title(loc="left")
    assert plan.reason.replace("_", " ") in texts.lower() or plan.reason in texts


def test_unreachable_figure_rejects_a_successful_plan(tmp_path):
    with pytest.raises(ValueError):
        viz.plot_unreachable(ARM, Q0, _plan(), TARGET, tmp_path / "x.png")


def test_drawing_does_not_touch_pyplot_state(tmp_path):
    import matplotlib.pyplot as plt

    before = plt.get_fignums()
    anim = viz.animate_plan(ARM, _plan(), TARGET, n_frames=6)
    viz.save_gif(anim, tmp_path / "a.gif", fps=5)
    viz.plot_snapshots(ARM, _plan(), TARGET, tmp_path / "s.png", n=3)
    assert plt.get_fignums() == before
