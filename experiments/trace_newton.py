"""Diagnostic: print the Newton iterates for a target near a workspace boundary.

Not a pass/fail experiment. It exists so that the explanation in docs/ENGINEERING_NOTE.md
(why the branch reached from the default initial guess is erratic near the inner boundary)
can be checked by anyone: the table shows theta2 of every iterate (unwrapped, as NMT keeps
it), how close that iterate is to a singular pose, and the step taken to reach it.

Run (from the repository root):
    python experiments/trace_newton.py [--boundary inner|outer] [--gap 0.1] [--rows 12]
The target is (r, 0) with r = min_reach + gap (inner) or max_reach - gap (outer), the same
construction as part C of experiments/run_ik_comparison.py.
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np

from manipulator_lab import PlanarArm, ik_newton


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--boundary", choices=("inner", "outer"), default="inner")
    parser.add_argument("--gap", type=float, default=0.1)
    parser.add_argument("--rows", type=int, default=12)
    args = parser.parse_args(argv)

    arm = PlanarArm()
    radius = arm.min_reach + args.gap if args.boundary == "inner" else arm.max_reach - args.gap
    result = ik_newton(arm, (radius, 0.0))
    print(f"arm l1={arm.l1}, l2={arm.l2}; target=({radius:.6g}, 0), {args.boundary} boundary, gap={args.gap}")
    print(f"converged={result.converged} iterations={result.iterations} branch={result.branch}")
    print(f"{'k':>3} {'theta2 (rad)':>14} {'|sin theta2|':>14} {'step error':>12}")
    for entry in (result.history or [])[: args.rows]:
        theta2 = float(np.asarray(entry["solution"], dtype=float)[1])
        step = entry["step_error"]
        step_text = "-" if step is None else f"{step:.4g}"
        print(f"{entry['iteration']:>3} {theta2:>+14.4f} {abs(math.sin(theta2)):>14.3e} {step_text:>12}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
