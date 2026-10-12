"""Experiment W4: joint-space trajectories (linear vs cubic rest-to-rest).

Question
    Do the linear and cubic trajectories satisfy their boundary conditions, do the closed-form
    cubic coefficients agree with the independent 4x4 solve (NMT gaussian_elimination), and
    what does each profile cost in peak velocity and acceleration?

Setup
    Default arm (l1 = 1.0, l2 = 0.8), T = 2.0 s, 201 time samples (odd, so t = T/2 is sampled),
    N random start/goal pairs with seed 0: start = random elbow-up configuration
    (theta1 uniform in [-pi, pi], theta2 uniform in [-pi, 0]); target = random reachable point
    (radius uniform in [0.25, 1.75], angle uniform); goal = analytic IK, elbow-up.

Definition of success (written BEFORE running, SPEC section 7 rule 5)
    C1  endpoint position error <= 1e-12 rad, both profiles, all trajectories.
    C2  cubic endpoint velocity <= 1e-12 rad/s; linear velocity equals (qf - q0)/T within 1e-12
        at every sample.
    C3  closed-form cubic coefficients vs the NMT 4x4 solve: max |difference| <=
        1e-9 * max(1, |coefficient|) over all joints.
    C4  peak velocity of cubic / linear = 1.5 within relative 1e-12 (joints with |delta| > 1e-6).
    C5  both profiles trace the same joint-space segment from q0 to qf: distance of every
        sample to that segment <= 1e-12 rad.
    C6  theta2 <= 1e-12 at every sample of every trajectory (elbow-up start and goal, so the
        path never crosses the stretched singularity theta2 = 0).
    C7  task-space end error |fk(q(T)) - target| <= 1e-9 m.
    A failed criterion is reported, not hidden: the script exits with status 1.

Reported without a criterion (descriptive)
    How far the end-effector path bends away from the straight chord between its two end
    points; how often the literal joint difference sends theta1 the long way round
    (|delta theta1| > pi); the smallest |sin theta2| and the largest cond(J) along the paths.

Outputs (in --out, default docs/results/)
    trajectory_profiles.png     position, velocity, acceleration of both profiles
    trajectory_example.png      one end-effector path with equal-time snapshots
    trajectory_summary.json     numbers, parameters, environment

Run (from the repository root):
    python experiments/run_trajectory.py [--n 1000] [--seed 0] [--out docs/results]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import env_info  # noqa: E402
from manipulator_lab import (  # noqa: E402
    PlanarArm,
    condition_from_theta2,
    fk,
    trajectory,
    workspace_boundary,
)

DURATION = 2.0
SAMPLES = 201
TOL_END = 1e-12
TOL_COEF = 1e-9
TOL_TASK = 1e-9

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3 of the reference palette
LINEAR_COLOR, CUBIC_COLOR = SERIES[0], SERIES[1]


def _style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_SECONDARY)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=3)
    ax.xaxis.label.set_color(TEXT_SECONDARY)
    ax.yaxis.label.set_color(TEXT_SECONDARY)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def _distance_to_segment(points, a, b):
    """Distance of each row of ``points`` to the segment a-b (a == b allowed)."""
    ab = b - a
    denom = float(ab @ ab)
    if denom == 0.0:
        return np.linalg.norm(points - a, axis=1)
    u = np.clip(((points - a) @ ab) / denom, 0.0, 1.0)
    return np.linalg.norm(points - (a + u[:, None] * ab), axis=1)


def run_cases(arm, n, seed):
    rng = np.random.default_rng(seed)
    t = trajectory.time_grid(DURATION, SAMPLES)
    rows = {k: [] for k in (
        "end_err_linear", "end_err_cubic", "cubic_end_vel", "linear_vel_err", "coef_err",
        "peak_ratio", "segment_dev", "theta2_max", "task_err", "chord_dev",
        "long_way", "min_abs_sin_theta2", "max_cond_on_path",
    )}
    for _ in range(n):
        radius, angle = rng.uniform(0.25, 1.75), rng.uniform(-math.pi, math.pi)
        target = (radius * math.cos(angle), radius * math.sin(angle))
        q0 = np.array([rng.uniform(-math.pi, math.pi), rng.uniform(-math.pi, 0.0)])
        plan_lin = trajectory.plan_to_target(arm, q0, target, DURATION, t, profile="linear")
        plan_cub = trajectory.plan_to_target(arm, q0, target, DURATION, t, profile="cubic")
        assert plan_lin.ok and plan_cub.ok, "targets are drawn inside the reachable annulus"
        lin, cub, qf = plan_lin.trajectory, plan_cub.trajectory, plan_cub.theta_target
        delta = qf - q0

        rows["end_err_linear"].append(max(np.max(np.abs(lin.q[0] - q0)), np.max(np.abs(lin.q[-1] - qf))))
        rows["end_err_cubic"].append(max(np.max(np.abs(cub.q[0] - q0)), np.max(np.abs(cub.q[-1] - qf))))
        rows["cubic_end_vel"].append(np.max(np.abs(cub.qd[[0, -1]])))
        rows["linear_vel_err"].append(np.max(np.abs(lin.qd - delta / DURATION)))

        for j in range(2):
            closed = trajectory.cubic_coefficients(q0[j], qf[j], DURATION)
            via_nmt = trajectory.cubic_coefficients_via_linear_system(q0[j], qf[j], DURATION)
            scale = max(1.0, float(np.max(np.abs(closed))))
            rows["coef_err"].append(float(np.max(np.abs(closed - via_nmt))) / scale)
            if abs(delta[j]) > 1e-6:
                v_cub = float(np.max(np.abs(cub.qd[:, j])))
                v_lin = float(np.max(np.abs(lin.qd[:, j])))
                rows["peak_ratio"].append(v_cub / v_lin)

        rows["segment_dev"].append(max(
            np.max(_distance_to_segment(lin.q, q0, qf)), np.max(_distance_to_segment(cub.q, q0, qf))
        ))
        rows["theta2_max"].append(max(np.max(lin.q[:, 1]), np.max(cub.q[:, 1])))
        rows["task_err"].append(float(np.linalg.norm(np.asarray(fk(arm, cub.q[-1])) - target)))

        ee = np.array([fk(arm, q) for q in cub.q])
        rows["chord_dev"].append(float(np.max(_distance_to_segment(ee, ee[0], ee[-1]))))
        rows["long_way"].append(bool(abs(delta[0]) > math.pi or abs(delta[1]) > math.pi))
        sin_min = float(np.min(np.abs(np.sin(cub.q[:, 1]))))
        rows["min_abs_sin_theta2"].append(sin_min)
        worst = cub.q[int(np.argmin(np.abs(np.sin(cub.q[:, 1])))), 1]
        rows["max_cond_on_path"].append(float(condition_from_theta2(arm, worst)))
    return {k: np.asarray(v) for k, v in rows.items()}


def plot_profiles(path):
    t = np.linspace(0.0, DURATION, 401)
    lin = trajectory.linear([0.0, 0.0], [1.0, 1.0], DURATION, t)
    cub = trajectory.cubic([0.0, 0.0], [1.0, 1.0], DURATION, t)
    fig, axes = plt.subplots(3, 1, figsize=(6.6, 7.0), dpi=150, sharex=True, facecolor=SURFACE)
    panels = [("q", "position (rad)"), ("qd", "velocity (rad/s)"), ("qdd", "acceleration (rad/s²)")]
    for ax, (field, label) in zip(axes, panels):
        _style_axes(ax)
        ax.plot(t, getattr(lin, field)[:, 0], color=LINEAR_COLOR, lw=2.0, label="linear")
        ax.plot(t, getattr(cub, field)[:, 0], color=CUBIC_COLOR, lw=2.0, label="cubic")
        ax.set_ylabel(label)
    axes[1].annotate("linear: velocity jumps\nfrom rest at t = 0 and t = T", xy=(0.0, 0.5),
                     xytext=(0.28, 0.2), fontsize=8, color=TEXT_SECONDARY,
                     arrowprops={"arrowstyle": "->", "color": TEXT_SECONDARY, "lw": 0.8})
    axes[1].annotate("cubic peak = 1.5 × linear", xy=(DURATION / 2, 0.75),
                     xytext=(1.2, 1.2), fontsize=8, color=TEXT_SECONDARY,
                     arrowprops={"arrowstyle": "->", "color": TEXT_SECONDARY, "lw": 0.8})
    axes[2].set_xlabel("time (s)")
    axes[0].set_title("Unit joint motion (Δ = 1 rad, T = 2 s)", fontsize=10, color=TEXT_PRIMARY, loc="left")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_example(arm, q0, target, path):
    t = trajectory.time_grid(DURATION, SAMPLES)
    plan = trajectory.plan_to_target(arm, q0, target, DURATION, t, profile="cubic")
    plan_lin = trajectory.plan_to_target(arm, q0, target, DURATION, t, profile="linear")
    ee = np.array([fk(arm, q) for q in plan.trajectory.q])
    stamps = np.linspace(0, SAMPLES - 1, 9).astype(int)
    outer, inner = workspace_boundary(arm)

    fig, ax = plt.subplots(figsize=(6.6, 5.2), dpi=150, facecolor=SURFACE)
    _style_axes(ax)
    ax.plot(outer[:, 0], outer[:, 1], color=TEXT_SECONDARY, lw=0.8)
    ax.plot(inner[:, 0], inner[:, 1], color=TEXT_SECONDARY, lw=0.8, ls="--")
    ax.plot(ee[:, 0], ee[:, 1], color=TEXT_SECONDARY, lw=1.0, zorder=2)
    ax.plot([ee[0, 0], ee[-1, 0]], [ee[0, 1], ee[-1, 1]], color=TEXT_SECONDARY, lw=1.0, ls=":",
            zorder=2, label="straight line in task space (for reference)")
    for name, color, traj, size in (("linear", LINEAR_COLOR, plan_lin.trajectory, 46),
                                    ("cubic", CUBIC_COLOR, plan.trajectory, 22)):
        pts = np.array([fk(arm, traj.q[i]) for i in stamps])
        ax.scatter(pts[:, 0], pts[:, 1], s=size, color=color, zorder=3, ec=SURFACE, lw=0.8,
                   label=f"{name}: end effector at equal time steps")
    for i, alpha in ((0, 0.9), (SAMPLES - 1, 0.9)):
        q = plan.trajectory.q[i]
        elbow = (arm.l1 * math.cos(q[0]), arm.l1 * math.sin(q[0]))
        tip = fk(arm, q)
        ax.plot([0, elbow[0], tip[0]], [0, elbow[1], tip[1]], color=TEXT_PRIMARY, lw=1.6, alpha=alpha)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Same path, different timing (start and goal arm poses drawn)", fontsize=10,
                 color=TEXT_PRIMARY, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main(argv=None) -> int:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=root / "docs" / "results")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    arm = PlanarArm()
    r = run_cases(arm, args.n, args.seed)

    ratio = r["peak_ratio"]
    criteria = {
        "C1_max_endpoint_error": float(max(r["end_err_linear"].max(), r["end_err_cubic"].max())),
        "C2_max_cubic_end_velocity": float(r["cubic_end_vel"].max()),
        "C2_max_linear_velocity_error": float(r["linear_vel_err"].max()),
        "C3_max_relative_coefficient_difference": float(r["coef_err"].max()),
        "C4_peak_velocity_ratio_min": float(ratio.min()),
        "C4_peak_velocity_ratio_max": float(ratio.max()),
        "C4_joints_compared": int(ratio.size),
        "C5_max_distance_to_joint_segment": float(r["segment_dev"].max()),
        "C6_max_theta2": float(r["theta2_max"].max()),
        "C7_max_task_space_end_error_m": float(r["task_err"].max()),
    }
    passed = {
        "C1": criteria["C1_max_endpoint_error"] <= TOL_END,
        "C2": criteria["C2_max_cubic_end_velocity"] <= TOL_END
        and criteria["C2_max_linear_velocity_error"] <= TOL_END,
        "C3": criteria["C3_max_relative_coefficient_difference"] <= TOL_COEF,
        "C4": bool(np.max(np.abs(ratio - 1.5)) / 1.5 <= 1e-12),
        "C5": criteria["C5_max_distance_to_joint_segment"] <= TOL_END,
        "C6": criteria["C6_max_theta2"] <= TOL_END,
        "C7": criteria["C7_max_task_space_end_error_m"] <= TOL_TASK,
    }
    descriptive = {
        "ee_path_max_deviation_from_chord_m": {
            "median": float(np.median(r["chord_dev"])),
            "p95": float(np.percentile(r["chord_dev"], 95)),
            "max": float(r["chord_dev"].max()),
        },
        "fraction_theta1_or_theta2_long_way_round": float(np.mean(r["long_way"])),
        "min_abs_sin_theta2_along_path": float(r["min_abs_sin_theta2"].min()),
        "max_cond_along_path": float(r["max_cond_on_path"].max()),
    }
    summary = {
        "experiment": "trajectory",
        "arm": {"l1": arm.l1, "l2": arm.l2},
        "duration_s": DURATION,
        "samples": SAMPLES,
        "trajectories": args.n,
        "seed": args.seed,
        "criteria_values": criteria,
        "criteria_pass": passed,
        "descriptive": descriptive,
        "environment": env_info.collect(),
    }

    plot_profiles(args.out / "trajectory_profiles.png")
    plot_example(arm, np.array([-0.6, -0.5]), (-0.3, 1.2), args.out / "trajectory_example.png")
    (args.out / "trajectory_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps({"trajectories": args.n, "criteria_values": criteria,
                      "criteria_pass": passed, "descriptive": descriptive}, indent=2))
    ok = all(passed.values())
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
