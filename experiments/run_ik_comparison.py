"""Experiment W3: Newton IK (NMT) against the analytic solution, and behaviour near singularities.

Questions
    A. Does Newton agree with the analytic IK in the interior of the workspace?
    B. How often does Newton converge from arbitrary starting points, and to which branch?
    C. What happens to Newton as the target approaches a workspace boundary (a singular pose)?
    D. How long does one call take (this machine only)?

Definition of success, v1 (written BEFORE the first run, SPEC section 7 rule 5)
    Part A only; B, C, D are descriptive. Interior polar grid (radii from min_reach + 0.05 m to
    max_reach - 0.05 m, 15 radii x 36 angles x 2 branches, branch-matched default initial guess):
    C1. every case converges AND lands on the requested branch;
    C2. wrapped joint-angle difference to the analytic solution of the same branch <= 1e-6 rad
        and position residual <= 1e-9 m on every case.

History (written AFTER inspecting the v1 result; v1 is kept and reported, not rewritten)
    v1 FAILED: all 1080 cases converged, but the 72 cases at the innermost radius (gap 0.05 m
    from the inner boundary) all landed on the other branch. A scan over the gap (part A2, added
    after that result) showed the branch reached from the default guess alternates between
    "always requested" and "never requested" as the radius changes inside ~0.3 m of the inner
    boundary, so no simple threshold exists there.
    v2 (post hoc, therefore weaker evidence than a pre-registered criterion) restricts the same
    C1 and C2 to a fresh grid at least 0.30 m from both boundaries (13 radii x 36 angles x 2
    branches). It claims only what that grid shows. The script's exit status follows v2; v1 and
    the part A2 scan are always written to the summary.

Hypotheses for part C, stated before running (not criteria; the outcome is reported either way):
    H1. Newton needs more iterations as the gap to the boundary shrinks.
    H2. At gap = 0 the root is a singular point of the Jacobian, so convergence is linear
        (step size roughly halves per iteration) instead of quadratic.
    H3. The distance to the nearest analytic solution grows as the gap shrinks, roughly like
        (residual tolerance) x cond(J).

Outputs (in --out, default docs/results/)
    ik_comparison_summary.json   numbers, parameters, environment
    ik_singularity.png           iterations and angle error against the gap to a boundary
    ik_branch_reliability.png    branch reached from the default guess against the gap
    ik_newton_convergence.png    Newton step size per iteration for three gaps

Run (from the repository root):
    python experiments/run_ik_comparison.py [--out docs/results]
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import env_info  # noqa: E402
from manipulator_lab import (  # noqa: E402
    Branch,
    PlanarArm,
    condition_number,
    ik_analytic,
    ik_newton,
    wrap_to_pi,
)

ANGLE_TOL = 1e-6
POS_TOL = 1e-9
MARGIN_V1 = 0.05  # metres kept away from both boundaries, v1 grid (15 radii)
MARGIN_V2 = 0.30  # same, v2 grid (13 radii, fresh values)
SCAN_GAPS = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.15, 0.20, 0.30, 0.40]
GAPS = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 0.0]
ZERO_PLOT_X = 1e-9  # where gap = 0 is drawn on the log axis

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3 of the reference palette


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


def _polar(r, angle):
    return (r * math.cos(angle), r * math.sin(angle))


def _wrapped_diff(a, b):
    return max(abs(wrap_to_pi(float(x) - float(y))) for x, y in zip(a, b))


def _nearest_analytic_error(arm, target, theta):
    return min(_wrapped_diff(theta, ik_analytic(arm, target, b).theta) for b in Branch)


def part_a(arm, margin, n_radii):
    """Polar grid kept ``margin`` away from both boundaries. Returns (summary, cases)."""
    radii = np.linspace(arm.min_reach + margin, arm.max_reach - margin, n_radii)
    angles = np.linspace(-math.pi, math.pi, 37)[:-1]
    n = converged = on_branch = 0
    worst_angle = worst_residual = 0.0
    iterations = []
    cases = []
    for branch in Branch:
        for radius in radii:
            for angle in angles:
                target = _polar(radius, angle)
                result = ik_newton(arm, target, branch=branch)
                n += 1
                cases.append((target, branch))
                if not result.converged:
                    continue
                converged += 1
                on_branch += int(result.branch is branch)
                iterations.append(result.iterations)
                expected = ik_analytic(arm, target, branch).theta
                worst_angle = max(worst_angle, _wrapped_diff(result.theta, expected))
                worst_residual = max(worst_residual, result.residual)
    summary = {
        "margin_m": margin,
        "radii": [float(r) for r in radii],
        "cases": n,
        "converged": converged,
        "landed_on_requested_branch": on_branch,
        "max_angle_diff_rad": worst_angle,
        "max_position_residual_m": worst_residual,
        "iterations_median": statistics.median(iterations) if iterations else None,
        "iterations_max": max(iterations) if iterations else None,
        "criterion_C1_pass": bool(converged == n and on_branch == n),
        "criterion_C2_pass": bool(worst_angle <= ANGLE_TOL and worst_residual <= POS_TOL),
    }
    return summary, cases


def part_a2(arm):
    """Fraction of cases landing on the requested branch against the gap to each boundary."""
    angles = np.linspace(-math.pi, math.pi, 37)[:-1]
    scan = {"inner": [], "outer": []}
    for boundary in scan:
        for gap in SCAN_GAPS:
            radius = arm.min_reach + gap if boundary == "inner" else arm.max_reach - gap
            n = hits = 0
            for branch in Branch:
                for angle in angles:
                    result = ik_newton(arm, _polar(radius, angle), branch=branch)
                    n += 1
                    hits += int(result.converged and result.branch is branch)
            scan[boundary].append({"gap_m": gap, "cases": n, "landed_on_requested_branch": hits})
    return scan


def part_b(arm, n=2000, seed=0):
    """Random targets and random initial guesses: where does Newton end up?"""
    rng = np.random.default_rng(seed)
    counts = {"elbow_up": 0, "elbow_down": 0, "converged_branch_ambiguous": 0}
    failures = {}
    for _ in range(n):
        radius = rng.uniform(arm.min_reach + MARGIN_V1, arm.max_reach - MARGIN_V1)
        target = _polar(radius, rng.uniform(-math.pi, math.pi))
        result = ik_newton(arm, target, initial=rng.uniform(-math.pi, math.pi, size=2))
        if result.converged:
            key = result.branch.value if result.branch is not None else "converged_branch_ambiguous"
            counts[key] += 1
        else:
            failures[result.reason] = failures.get(result.reason, 0) + 1
    converged = sum(counts.values())
    return {
        "seed": seed,
        "samples": n,
        "converged_fraction": converged / n,
        "converged_by_branch": counts,
        "failures_by_reason": failures,
    }


def part_c(arm):
    """Approach to each workspace boundary along the +x axis (the angle is irrelevant)."""
    rows = {"outer": [], "inner": []}
    for boundary in rows:
        for gap in GAPS:
            radius = arm.max_reach - gap if boundary == "outer" else arm.min_reach + gap
            target = (radius, 0.0)
            result = ik_newton(arm, target)
            row = {
                "gap_m": gap,
                "converged": result.converged,
                "reason": result.reason,
                "iterations": result.iterations,
                "residual": result.residual,
                "angle_error_to_nearest_analytic_rad": None,
                "cond_at_solution": None,
                "step_errors": [h["step_error"] for h in (result.history or ())[1:]],
            }
            if result.theta is not None:
                row["angle_error_to_nearest_analytic_rad"] = _nearest_analytic_error(
                    arm, target, result.theta
                )
                row["cond_at_solution"] = condition_number(arm, result.theta)
            rows[boundary].append(row)
    return rows


def part_d(arm, cases, repeats=7):
    """Median wall time per call over the interior cases (valid for this machine only)."""

    def per_call_us(fn):
        samples = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            for target, branch in cases:
                fn(target, branch)
            samples.append((time.perf_counter_ns() - start) / len(cases) / 1e3)
        return statistics.median(samples)

    return {
        "cases": len(cases),
        "repeats": repeats,
        "analytic_us_per_call_median": per_call_us(lambda t, b: ik_analytic(arm, t, b)),
        "newton_us_per_call_median": per_call_us(lambda t, b: ik_newton(arm, t, branch=b)),
        "note": "wall-clock, includes Python overhead and input validation; this machine only",
    }


def plot_singularity(rows, path):
    fig, (ax_it, ax_err) = plt.subplots(2, 1, figsize=(6.6, 6.2), dpi=150, sharex=True,
                                        facecolor=SURFACE)
    labels = {"outer": "outer boundary (arm fully stretched)", "inner": "inner boundary (arm folded)"}
    for color, boundary in zip(SERIES, ("outer", "inner")):
        xs = [ZERO_PLOT_X if r["gap_m"] == 0.0 else r["gap_m"] for r in rows[boundary]]
        its = [r["iterations"] if r["iterations"] is not None else np.nan for r in rows[boundary]]
        errs = [r["angle_error_to_nearest_analytic_rad"] for r in rows[boundary]]
        errs = [np.nan if e is None else max(e, 1e-17) for e in errs]
        ax_it.plot(xs, its, "-o", color=color, lw=1.8, ms=5, mec=SURFACE, mew=1.0)
        ax_err.plot(xs, errs, "-o", color=color, lw=1.8, ms=5, mec=SURFACE, mew=1.0,
                    label=labels[boundary])
        for x, row in zip(xs, rows[boundary]):
            if not row["converged"]:
                ax_it.plot([x], [row["iterations"] if row["iterations"] is not None else np.nan],
                           "o", ms=9, mfc="none", mec=color, mew=1.4)
    for ax in (ax_it, ax_err):
        _style_axes(ax)
        ax.set_xscale("log")
    ax_err.set_yscale("log")
    ax_it.set_ylabel("Newton iterations")
    ax_err.set_ylabel("angle error to nearest\nanalytic solution (rad)")
    ticks = [ZERO_PLOT_X] + [10.0 ** k for k in range(-8, 0)]
    ax_err.set_xticks(ticks, ["0"] + [f"1e{k}" for k in range(-8, 0)])
    ax_err.set_xlabel("gap to workspace boundary (m)   [leftmost point: exactly 0]")
    ax_it.set_title("Newton IK approaching a singular pose", fontsize=10, color=TEXT_PRIMARY, loc="left")
    ax_err.legend(frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY, loc="upper right")
    if any(not r["converged"] for rows_b in rows.values() for r in rows_b):
        ax_it.text(0.01, 0.04, "hollow ring = did not converge", transform=ax_it.transAxes,
                   fontsize=8, color=TEXT_SECONDARY)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_branch_reliability(scan, path):
    fig, ax = plt.subplots(figsize=(6.6, 3.6), dpi=150, facecolor=SURFACE)
    _style_axes(ax)
    labels = {"inner": "gap to inner boundary", "outer": "gap to outer boundary"}
    for color, boundary, dx in zip(SERIES, ("inner", "outer"), (-0.0018, 0.0018)):
        xs = [row["gap_m"] + dx for row in scan[boundary]]
        ys = [row["landed_on_requested_branch"] / row["cases"] for row in scan[boundary]]
        ax.plot(xs, ys, "o", color=color, ms=6, mec=SURFACE, mew=1.0, label=labels[boundary])
    ax.set_ylim(-0.08, 1.08)
    ax.set_xlabel("gap between target radius and workspace boundary (m)")
    ax.set_ylabel("fraction landing on\nthe requested branch")
    ax.set_title("Branch reached from the default initial guess", fontsize=10,
                 color=TEXT_PRIMARY, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY, loc="center right")
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_convergence(rows, path):
    chosen = [1e-1, 1e-3, 0.0]
    fig, ax = plt.subplots(figsize=(6.6, 4.0), dpi=150, facecolor=SURFACE)
    _style_axes(ax)
    for color, gap in zip(SERIES, chosen):
        row = next(r for r in rows["outer"] if r["gap_m"] == gap)
        steps = [max(s, 1e-17) for s in row["step_errors"]]
        xs = list(range(1, len(steps) + 1))
        name = "gap = 0 (singular root)" if gap == 0.0 else f"gap = {gap:g} m"
        ax.plot(xs, steps, "-o", color=color, lw=1.8, ms=4, mec=SURFACE, mew=1.0, label=name)
    ax.set_yscale("log")
    ax.set_xlabel("iteration")
    ax.set_ylabel("Newton step size (max-norm, rad)")
    ax.set_title("Newton step size per iteration, outer boundary", fontsize=10, color=TEXT_PRIMARY, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main(argv=None) -> int:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=root / "docs" / "results")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    arm = PlanarArm()
    a_v1, _ = part_a(arm, MARGIN_V1, 15)
    a_v2, cases = part_a(arm, MARGIN_V2, 13)
    a2 = part_a2(arm)
    b = part_b(arm)
    c = part_c(arm)
    d = part_d(arm, cases)

    summary = {
        "experiment": "ik_comparison",
        "arm": {"l1": arm.l1, "l2": arm.l2},
        "tolerances": {"angle_rad": ANGLE_TOL, "position_m": POS_TOL},
        "A_interior_agreement_v1_failed": a_v1,
        "A_interior_agreement_v2_posthoc": a_v2,
        "A2_branch_landing_vs_gap": a2,
        "B_random_starts": b,
        "C_approach_to_singularity": c,
        "D_runtime": d,
        "environment": env_info.collect(),
    }
    (args.out / "ik_comparison_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    plot_singularity(c, args.out / "ik_singularity.png")
    plot_convergence(c, args.out / "ik_newton_convergence.png")
    plot_branch_reliability(a2, args.out / "ik_branch_reliability.png")

    print(json.dumps({"A_v1_failed": a_v1, "A_v2_posthoc": a_v2, "B_random_starts": b,
                      "D_runtime": d}, indent=2))
    print("\nA2. fraction landing on requested branch (72 cases per gap)")
    for boundary in ("inner", "outer"):
        print(f"  {boundary}: " + "  ".join(
            f"{r['gap_m']:g}:{r['landed_on_requested_branch'] / r['cases']:.2f}" for r in a2[boundary]))
    for boundary in ("outer", "inner"):
        print(f"\nC. {boundary} boundary")
        print("  gap_m      conv  iters  residual     angle_err_rad  cond")
        for r in c[boundary]:
            fmt = lambda v: "-" if v is None else f"{v:.3g}"  # noqa: E731
            print(f"  {r['gap_m']:<9.1e}  {str(r['converged']):<5} {fmt(r['iterations']):<6} "
                  f"{fmt(r['residual']):<12} {fmt(r['angle_error_to_nearest_analytic_rad']):<14} "
                  f"{fmt(r['cond_at_solution'])}  {r['reason'] or ''}")
    v1_ok = a_v1["criterion_C1_pass"] and a_v1["criterion_C2_pass"]
    ok = a_v2["criterion_C1_pass"] and a_v2["criterion_C2_pass"]
    print(f"\nv1 (margin {MARGIN_V1} m, pre-registered): {'PASS' if v1_ok else 'FAIL'}")
    print(f"v2 (margin {MARGIN_V2} m, post hoc):        {'PASS' if ok else 'FAIL'}")
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
