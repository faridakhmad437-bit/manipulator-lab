"""Experiment W2: conditioning of the Jacobian over the workspace.

Question
    How does cond(J) vary over the planar 2-link workspace, and does it depend on
    theta2 only, as derived in SPEC section 3.4?

Definition of success (written BEFORE running, SPEC section 7 rule 5)
    1. Over every reachable finite grid cell, the SVD-based ``condition_number`` at the
       IK solution agrees with the closed-form ``condition_from_theta2`` with maximum
       relative difference <= 1e-9.
    2. cond(J) >= 1 on every reachable cell.
    3. Results are reported with the grid resolution and the environment used.
    A failed criterion is reported, not hidden: the script exits with status 1.

Outputs (in --out, default docs/results/)
    conditioning_map.png         cond(J) over the workspace (log colour scale)
    conditioning_vs_theta2.png   cond(J) against theta2
    conditioning_summary.json    numbers, parameters, environment

Run (from the repository root):
    python experiments/run_conditioning.py [--resolution 401] [--out docs/results]
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
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402

import env_info  # noqa: E402
from manipulator_lab import (  # noqa: E402
    Branch,
    PlanarArm,
    condition_from_theta2,
    condition_map,
    condition_number,
    ik_analytic,
    workspace_boundary,
)

REL_TOL = 1e-9

# Reference palette (dataviz skill): sequential blue ramp, light surface, text tokens.
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SEQ_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SERIES_1 = "#2a78d6"


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


def _theta2_of_cell(arm, x, y):
    result = ik_analytic(arm, (x, y), Branch.ELBOW_DOWN)
    return float(result.theta[1])


def validate_against_closed_form(arm, cmap):
    """Criteria 1 and 2. Returns (max relative difference, min cond, n cells compared)."""
    worst, smallest, compared = 0.0, math.inf, 0
    for iy, y in enumerate(cmap.y):
        for ix, x in enumerate(cmap.x):
            if not cmap.reachable[iy, ix]:
                continue
            value = cmap.cond[iy, ix]
            smallest = min(smallest, value)
            if not math.isfinite(value):
                continue
            closed = condition_from_theta2(arm, _theta2_of_cell(arm, x, y))
            if not math.isfinite(closed):
                worst = math.inf
                continue
            worst = max(worst, abs(value - closed) / closed)
            compared += 1
    return worst, smallest, compared


def plot_map(arm, cmap, path):
    ramp = LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)
    ramp.set_bad(SURFACE)
    fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=150, facecolor=SURFACE)
    _style_axes(ax)
    extent = (cmap.x[0], cmap.x[-1], cmap.y[0], cmap.y[-1])
    image = ax.imshow(
        np.ma.masked_invalid(cmap.cond),
        origin="lower",
        extent=extent,
        cmap=ramp,
        norm=LogNorm(vmin=1.0, vmax=100.0),
        interpolation="nearest",
    )
    outer, inner = workspace_boundary(arm)
    ax.plot(outer[:, 0], outer[:, 1], color=TEXT_SECONDARY, lw=0.8)
    ax.plot(inner[:, 0], inner[:, 1], color=TEXT_SECONDARY, lw=0.8)
    ax.annotate(
        f"r = {arm.max_reach:.1f} m", (0.0, arm.max_reach), xytext=(0, 5),
        textcoords="offset points", ha="center", fontsize=8, color=TEXT_SECONDARY,
    )
    ax.annotate(
        f"r = {arm.min_reach:.1f} m", (0.0, arm.min_reach), xytext=(0, 4),
        textcoords="offset points", ha="center", fontsize=8, color=TEXT_PRIMARY,
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(
        f"cond(J) over the workspace  (l1 = {arm.l1} m, l2 = {arm.l2} m)",
        fontsize=10, color=TEXT_PRIMARY, loc="left",
    )
    bar = fig.colorbar(image, ax=ax, extend="max", fraction=0.046, pad=0.03)
    bar.set_label("cond(J), log scale (capped at 100)", color=TEXT_SECONDARY, fontsize=9)
    bar.ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    bar.outline.set_edgecolor(TEXT_SECONDARY)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_vs_theta2(arm, theta2, cond, path):
    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=150, facecolor=SURFACE)
    _style_axes(ax)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.plot(theta2, cond, color=SERIES_1, lw=1.8)
    k = int(np.argmin(cond))
    ax.plot([theta2[k]], [cond[k]], "o", ms=6, color=SERIES_1, mec=SURFACE, mew=1.5)
    ax.annotate(
        f"minimum {cond[k]:.2f} at \u03b8\u2082 = {math.degrees(theta2[k]):.1f}\u00b0",
        (theta2[k], cond[k]), xytext=(1.5, 30.0), textcoords="data",
        fontsize=8, color=TEXT_PRIMARY, ha="left",
        arrowprops={"arrowstyle": "-", "color": TEXT_SECONDARY, "lw": 0.8},
    )
    ax.set_yscale("log")
    ax.set_xlim(0.0, math.pi)
    ax.set_xticks([0, math.pi / 2, math.pi], ["0", "\u03c0/2", "\u03c0"])
    ax.set_xlabel("\u03b8\u2082 (rad)   [0 and \u03c0 are singular poses]")
    ax.set_ylabel("cond(J)")
    ax.set_title(
        f"cond(J) against \u03b8\u2082  (l1 = {arm.l1} m, l2 = {arm.l2} m)",
        fontsize=10, color=TEXT_PRIMARY, loc="left",
    )
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def main(argv=None) -> int:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--resolution", type=int, default=401)
    parser.add_argument("--out", type=Path, default=root / "docs" / "results")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    arm = PlanarArm()
    cmap = condition_map(arm, resolution=args.resolution)
    worst, smallest, compared = validate_against_closed_form(arm, cmap)

    finite = cmap.cond[cmap.reachable]
    finite = finite[np.isfinite(finite)]
    theta2 = np.linspace(1e-3, math.pi - 1e-3, 1200)
    curve = np.array([condition_from_theta2(arm, t) for t in theta2])
    samples = [math.pi / 2, 1.0, 0.5, 0.1, 0.01, 1e-3, 1e-6]
    table = [
        {
            "theta2_rad": t,
            "cond_svd": condition_number(arm, (0.4, t)),
            "cond_closed_form": condition_from_theta2(arm, t),
        }
        for t in samples
    ]

    criteria = {
        "max_rel_diff_svd_vs_closed_form": worst,
        "rel_tolerance": REL_TOL,
        "cells_compared": compared,
        "min_cond_on_reachable_cells": smallest,
        "criterion_1_pass": bool(worst <= REL_TOL),
        "criterion_2_pass": bool(smallest >= 1.0),
    }
    summary = {
        "experiment": "conditioning",
        "arm": {"l1": arm.l1, "l2": arm.l2},
        "grid_resolution": args.resolution,
        "grid_extent_m": float(cmap.x[-1]),
        "reachable_cells": int(cmap.reachable.sum()),
        "finite_cond_cells": int(finite.size),
        "fraction_of_reachable_cells": {
            "cond_le_10": float(np.mean(finite <= 10.0)),
            "cond_gt_100": float(np.mean(finite > 100.0)),
        },
        "min_cond_over_theta2_curve": float(curve.min()),
        "theta2_at_min_rad": float(theta2[int(np.argmin(curve))]),
        "samples": table,
        "criteria": criteria,
        "environment": env_info.collect(),
    }

    plot_map(arm, cmap, args.out / "conditioning_map.png")
    plot_vs_theta2(arm, theta2, curve, args.out / "conditioning_vs_theta2.png")
    (args.out / "conditioning_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )

    print(json.dumps({k: summary[k] for k in ("grid_resolution", "criteria")}, indent=2))
    ok = criteria["criterion_1_pass"] and criteria["criterion_2_pass"]
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
