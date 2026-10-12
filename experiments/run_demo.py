"""Demo W5: end-to-end run (reachability -> IK -> cubic trajectory -> animation) and one failure.

Question
    Does the whole pipeline run from one command, does the drawn arm agree with forward
    kinematics frame by frame, and is an unreachable target reported instead of drawn?

Scenario
    Default arm (l1 = 1.0, l2 = 0.8), start q0 = (-0.5, -0.3) rad, target (-0.6, 1.0) m,
    T = 2.0 s, cubic rest-to-rest, elbow-up, 60 animation frames at 20 fps (+ 20 held frames).
    Failure case: target (1.6, 1.2) m, 2.0 m from the shoulder (max reach 1.8 m).

Definition of success (written BEFORE running, SPEC section 7 rule 5)
    D1  first rendered frame: tip within 1e-12 m of fk(q0); last rendered frame: tip within
        1e-9 m of the target. Positions are read back from the matplotlib artist, not from
        the arrays used to build it.
    D2  in every rendered frame the links are rigid: ||elbow|| = l1 and ||tip - elbow|| = l2,
        each within 1e-12 m.
    D3  files: the GIF exists, is a valid GIF with 2..60 frames (identical consecutive frames
        may be merged by the encoder) and is at most 2 MB; both PNGs start with the PNG
        signature and are non-empty.
    D4  the unreachable target gives ok = False, reason = target_beyond_max_reach and no
        trajectory.
    A failed criterion is reported, not hidden: the script exits with status 1. Looking at
    the frames is a separate manual step and is not part of these criteria.

Outputs (in --out, default docs/results/)
    demo_reach.gif, demo_poses.png, demo_unreachable.png, demo_summary.json

Run (from the repository root):
    python experiments/run_demo.py [--out docs/results]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

import env_info
from manipulator_lab import BEYOND_MAX_REACH, PlanarArm, fk, trajectory, viz
from PIL import Image

DURATION = 2.0
SAMPLES = 201
N_FRAMES = 60
FPS = 20
HOLD = 20
Q0 = np.array([-0.5, -0.3])
TARGET = (-0.6, 1.0)
UNREACHABLE = (1.6, 1.2)
TOL_FIRST = 1e-12
TOL_LAST = 1e-9
TOL_RIGID = 1e-12
MAX_GIF_BYTES = 2_000_000
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def main(argv=None) -> int:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=root / "docs" / "results")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    arm = PlanarArm()
    t = trajectory.time_grid(DURATION, SAMPLES)
    plan = trajectory.plan_to_target(arm, Q0, TARGET, DURATION, t, profile="cubic")
    if not plan.ok:
        print("RESULT: FAIL (reachable demo target was rejected:", plan.reason, ")")
        return 1

    anim = viz.animate_plan(arm, plan, TARGET, n_frames=N_FRAMES, title="Cubic trajectory to the target")
    first_err = last_err = rigid_err = 0.0
    for i in range(N_FRAMES):
        anim.set_frame(i)
        pts = np.column_stack(anim.arm_line.get_data())
        rigid_err = max(rigid_err, abs(math.hypot(*pts[1]) - arm.l1),
                        abs(math.hypot(*(pts[2] - pts[1])) - arm.l2))
        if i == 0:
            first_err = float(np.hypot(*(pts[2] - np.asarray(fk(arm, Q0)))))
        if i == N_FRAMES - 1:
            last_err = float(np.hypot(*(pts[2] - np.asarray(TARGET))))

    gif_path = args.out / "demo_reach.gif"
    poses_path = args.out / "demo_poses.png"
    fail_path = args.out / "demo_unreachable.png"
    viz.save_gif(anim, gif_path, fps=FPS, hold=HOLD)
    viz.plot_snapshots(arm, plan, TARGET, poses_path, n=6)
    bad_plan = trajectory.plan_to_target(arm, Q0, UNREACHABLE, DURATION, t)
    viz.plot_unreachable(arm, Q0, bad_plan, UNREACHABLE, fail_path)

    with Image.open(gif_path) as gif:
        gif_format, gif_frames, gif_size_px = gif.format, gif.n_frames, list(gif.size)
    gif_bytes = gif_path.stat().st_size
    png_ok = all(p.stat().st_size > 0 and p.read_bytes()[:8] == PNG_SIGNATURE
                 for p in (poses_path, fail_path))

    values = {
        "D1_first_frame_tip_error_m": first_err,
        "D1_last_frame_tip_error_m": last_err,
        "D2_max_link_length_error_m": float(rigid_err),
        "D3_gif_format": gif_format,
        "D3_gif_frames_after_encoding": gif_frames,
        "D3_gif_bytes": gif_bytes,
        "D3_gif_size_px": gif_size_px,
        "D3_pngs_valid": png_ok,
        "D4_unreachable_ok": bad_plan.ok,
        "D4_unreachable_reason": bad_plan.reason,
        "D4_unreachable_has_trajectory": bad_plan.trajectory is not None,
    }
    passed = {
        "D1": first_err <= TOL_FIRST and last_err <= TOL_LAST,
        "D2": rigid_err <= TOL_RIGID,
        "D3": gif_format == "GIF" and 2 <= gif_frames <= N_FRAMES
        and 0 < gif_bytes <= MAX_GIF_BYTES and png_ok,
        "D4": (not bad_plan.ok) and bad_plan.reason == BEYOND_MAX_REACH
        and bad_plan.trajectory is None,
    }
    summary = {
        "experiment": "demo",
        "arm": {"l1": arm.l1, "l2": arm.l2},
        "scenario": {"q0": Q0.tolist(), "target": list(TARGET), "duration_s": DURATION,
                     "profile": "cubic", "branch": "elbow_up", "animation_frames": N_FRAMES,
                     "fps": FPS, "hold_frames": HOLD, "unreachable_target": list(UNREACHABLE)},
        "criteria_values": values,
        "criteria_pass": passed,
        "environment": env_info.collect(),
    }
    (args.out / "demo_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"criteria_values": values, "criteria_pass": passed}, indent=2))
    ok = all(passed.values())
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
