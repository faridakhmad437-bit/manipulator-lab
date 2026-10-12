# Manipulator Lab

Planar 2-link manipulator: forward kinematics, analytic inverse kinematics, reachability checks, and (planned) Newton-based IK, singularity analysis, and trajectory planning on top of [Numerical Method Toolkit](https://github.com/faridakhmad437-bit/Numerical-Method-Toolkit) (pinned to `v1.0.0`).

Full design: [SPEC.md](SPEC.md).

## Status

| Component | Status |
|---|---|
| Arm model, FK, analytic IK (both branches), reachability | Implemented, tests pass (groups A and B) |
| Jacobian, conditioning, workspace map | Planned (week 2) |
| Newton IK via NMT and comparison | Planned (week 3) |
| Trajectories (linear, cubic) | Planned (week 4) |
| Visualization, demo, engineering note | Planned (weeks 5-6) |

No performance or accuracy claims are made until they are measured and recorded under `docs/results/`.

## Conventions

- Units: metres, radians, seconds. World frame at the shoulder, x right, y up, CCW positive.
- `theta1` is absolute; `theta2` is relative to link 1.
- **Elbow-up = negative `theta2`**, elbow-down = positive `theta2`.
- Defaults: `l1 = 1.0 m`, `l2 = 0.8 m` (reach `0.2 <= r <= 1.8 m`).
- Unreachable targets return `reachable=False` with a `reason`; invalid inputs raise `ValueError`.

## Quick start (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest
ruff check .
```

```python
from manipulator_lab import PlanarArm, Branch, fk, ik_analytic

arm = PlanarArm()
result = ik_analytic(arm, (1.0, 0.5), Branch.ELBOW_UP)
print(result.reachable, result.theta, fk(arm, result.theta))
```

## Layout

```
src/manipulator_lab/   arm.py, kinematics.py (more modules per SPEC section 4)
tests/                 test_kinematics.py (group A), test_reachability.py (group B)
experiments/           reproducible experiment scripts (planned)
docs/results/          recorded plots, tables, environment metadata (planned)
```
