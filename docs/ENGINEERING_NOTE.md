# Why Newton IK degrades near singularities (planar 2-link arm)

**Scope.** One arm (`l1 = 1.0 m`, `l2 = 0.8 m`), Newton's method from NMT v1.0.0 (`newton_system`) on `F(q) = fk(q) - target` with the analytic Jacobian. Every number below is read from a file under `docs/results/` produced on the author's machine (Windows 11, Python 3.14.3, NumPy 2.5.4), except where a line says otherwise. Statements are tagged **Derived** (algebra, checked against data), **Measured** (read from a result file), or **Interpretation** (consistent with the data, not isolated by a dedicated test).

## 1. The Jacobian gets worse in a predictable way

`det J = l1 l2 sin(theta2)`, so J is singular exactly when the arm is fully stretched (`theta2 = 0`) or fully folded (`theta2 = pi`). The condition number depends on `theta2` only:

`cond(J) = sigma1^2 / D`, with `F = l1^2 + 2 l2^2 + 2 l1 l2 cos(theta2)`, `D = l1 l2 |sin(theta2)|`, `sigma1^2 = (F + sqrt(F^2 - 4 D^2)) / 2`.

**Measured** (`conditioning_summary.json`): closed form and SVD agree to a relative `4.7e-14` on all 102,564 reachable grid cells. Near `theta2 = 0`, `cond` is `485.0`, `4850.0`, `4.85e6` at `theta2 = 1e-2, 1e-3, 1e-6`: it scales as `F0 / (l1 l2 |theta2|) = 4.85 / |theta2|`. Away from the boundaries the map is benign: 94.2% of cells have `cond <= 10`, and only 0.055% exceed 100.

**Derived, then checked.** A target at distance `g` from a workspace boundary forces `theta2` to be small, and that gives a square-root law:

- Outer (stretched): `theta2 = sqrt(2 (l1 + l2) g / (l1 l2))`, so `cond ~ 2.286 / sqrt(g)`.
- Inner (folded, `phi = pi - theta2`): `phi = sqrt(2 (l1 - l2) g / (l1 l2))`, so `cond ~ 1.202 / sqrt(g)`.

`ik_comparison_summary.json`, part C, `cond` at the converged solution:

| gap g (m) | outer measured | outer `2.286/sqrt(g)` | inner measured | inner `1.202/sqrt(g)` |
|---|---|---|---|---|
| 1e-2 | 22.77 | 22.86 | 11.87 | 12.02 |
| 1e-4 | 228.6 | 228.6 | 120.2 | 120.2 |
| 1e-6 | 2286 | 2286 | 1202 | 1202 |
| 1e-8 | 2.286e4 | 2.286e4 | 1.202e4 | 1.202e4 |

The relative deviation shrinks in proportion to `g` (about `-0.42 g` outer, `-1.3 g` inner), as expected for a leading-order expansion. The stretched boundary is worse than the folded one by `2.286 / 1.202 = 1.90` at the same gap.

## 2. What Newton does with that

**Interior (Measured).** Quadratic convergence: 936 of 936 targets at least 0.3 m from both boundaries converged on the requested branch, median 5 and maximum 6 iterations, maximum angle difference to the analytic solution `8.9e-16 rad`, maximum position residual `7.8e-16 m`.

**Approaching the stretched boundary (Measured).** Iterations grow slowly: `6, 8, 9, 11, 12, 14, 16, 17` for `g = 1e-1 ... 1e-8`, about 1.5 per decade of `g`. The step-error trace shows a first phase where the error shrinks by a factor of about 0.45 to 0.5 per iteration, followed by a quadratic collapse. *Interpretation:* the linear phase lasts until the error is of the order of `theta2 ~ sqrt(g)`, which needs roughly `log2(1/sqrt(g))` halvings, i.e. 1.66 iterations per decade. This is close to the measured 1.5 but was not tested further.

**At the singular root, `g = 0` (Measured).** 28 iterations instead of 6. In the last phase the step error halves each iteration (`8.92e-5, 4.46e-5, 2.23e-5, 1.12e-5, ...`): linear convergence with ratio 0.5. *Interpretation:* the radial residual is quadratic in `theta2` there, so the root has multiplicity 2, and Newton's method converges linearly with ratio `(m - 1)/m = 1/2` on a double root. The solver still reports success: position residual `0`, angle error `1.3e-9 rad` (tolerance `1e-6`). The `cond` reported at that solution, `3.6e9`, is just `4.85 / 1.33e-9`: it measures how close to `theta2 = 0` the iteration stopped, not a property of the target.

**Residual versus angle error (Measured).** A small residual does not mean accurate angles near a singularity. On the folded side the angle error against the analytic solution grows, not strictly monotonically at the start, from `8.0e-15` (`g = 0.1`, `cond = 3.4`) to `1.0e-11` (`g = 1e-8`, `cond = 1.2e4`) while the position residual never exceeds `1.4e-14` m. On the stretched side the error is noisy (`8.9e-16`, `3.2e-13`, `3.6e-15` at `g = 0.1, 1e-7, 1e-8`), so no monotone law is claimed.

## 3. The folded boundary breaks branch selection, not accuracy

`default_initial_guess` starts at `theta2 = -pi/2` (elbow-up). Near the folded boundary the branch Newton ends on is erratic.

**Measured** (`ik_comparison_summary.json`, part A2, 72 targets per radius): the fraction landing on the requested branch is either 72/72 or 0/72 at each radius, not a gradual change. 72/72 at `g = 0.005, 0.02, 0.06, 0.12, 0.2, 0.3, 0.4`; 0/72 at `g = 0.01, 0.03 to 0.05, 0.07 to 0.1, 0.15`. At the stretched boundary it is 72/72 at every gap tested. The pre-registered interior check (margin 0.05 m) therefore failed: 1080 of 1080 converged, but only 1008 on the requested branch, maximum angle difference 1.13 rad. The post-hoc margin of 0.3 m (936 of 936) was chosen after seeing this failure and is labelled as such in `SPEC.md`.

**Mechanism, traced** (`python experiments/trace_newton.py`, target `(0.3, 0)`, `g = 0.1`; the step errors match those stored in the Windows result file):

| k | theta2 of iterate (rad) | abs sin(theta2) | step |
|---|---|---|---|
| 0 | -1.5708 | 1.0 | - |
| 1 | -3.1406 | 1.0e-3 | 1.57 |
| 2 | -31.014 | 0.39 | 112.9 |
| 3 | -20.957 | 0.86 | 10.06 |

The first step lands 1e-3 from the folded singularity (`theta2 = -pi`). A Newton step is `J^-1 F`, which scales like `1 / sin(theta2)`, so the next step is 112.9 rad. NMT keeps iterates unwrapped, so the iterate ends up near `theta2 = -31`, and which of the two branches the iteration finally converges to is decided by which `2 pi` sheet it lands on (here it ends elbow-down although elbow-up was requested). *Interpretation:* because a tiny change in the target changes how close the first iterate gets to `-pi`, the landing sheet changes abruptly with the radius, which fits the all-or-nothing pattern above. Only this trace for `g = 0.1` and the matching Windows step errors were examined; the pattern at the other radii was not traced.

**Measured, global behaviour.** From 2000 random initial guesses (seed 0): 88% converge (837 elbow-up, 923 elbow-down), 240 hit the iteration limit. The failures were not analysed further.

## 4. What this means in the code

- `ik_newton` never raises for numerical failures; it returns `reason` (`max_iterations`, `singular_jacobian`, `non_finite_value`, or the unreachable reasons). Tests cover each.
- Pass `initial=` explicitly when the target is within about 0.3 m of the inner boundary, or when the requested branch matters. `default_initial_guess` is only dependable in the interior and near the stretched boundary.
- **Not implemented and not tested here** (reasonable next steps): seeding Newton with the analytic solution on the desired branch; damped steps or a line search to stop the 112 rad jump; damped least squares for the Jacobian inverse. The analytic IK has no such problem and stays the baseline.

## 5. Limits

One arm geometry; part C uses targets on the x axis only; the iteration-count explanation in section 2 and the mechanism in section 3 are interpretations backed by one trace, not proofs. Runtime was measured (analytic about 6 to 8 microseconds per call, Newton about 230, two runs on this machine) but varies by roughly 30% between runs, so no speed ratio is claimed.

## 6. Where each number comes from

| Claim | File |
|---|---|
| cond closed form vs SVD, cell fractions, `4.85/theta2` samples | `docs/results/conditioning_summary.json` |
| interior agreement, v1 failure, A2 landing table, part C iterations/errors/cond, random starts | `docs/results/ik_comparison_summary.json` |
| convergence plots, branch scan, singularity approach | `docs/results/ik_newton_convergence.png`, `ik_branch_reliability.png`, `ik_singularity.png` |
| iterate trace in section 3 | `python experiments/trace_newton.py` (also reproduces the `1.57, 112.9, 10.06` step errors in the JSON) |
| pre-registered criteria and their history | `SPEC.md` section 7 and the docstring of `experiments/run_ik_comparison.py` |
