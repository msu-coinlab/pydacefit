# Surrogate benchmark

Which DACE **regression trend × correlation kernel** best surrogates classical
optimization test functions, across dimensions?

```bash
pyclawd python benchmark/surrogate_benchmark.py                 # sphere/rosenbrock/rastrigin/ackley, d=2,5,10
pyclawd python benchmark/surrogate_benchmark.py --quick         # fast preset
pyclawd python benchmark/surrogate_benchmark.py --dims 2 8 --funcs ackley
```

Each config fits DACE (with ARD theta-optimization) on a uniform sample
(`n_train = 12·d`) and is scored on a held-out test set by **NRMSE** = RMSE /
std(test). `0` = perfect surrogate, `~1` = no better than predicting the mean.
Train/test data are fixed per function/dim, so the comparison is fair.

## Findings (d = 2, 5, 10; n_test = 1000)

Winners tally (best config per function/dim, 12 cells):

| count | config |
|---|---|
| 3× | `quadratic + expg` |
| 3× | `constant + expg` |
| 2× | `quadratic + spline` |
| 1× each | `linear + gauss`, `linear + expg`, `quadratic + exp`, `quadratic + spherical` |

**1. `expg` (exponential-with-power kernel) is the standout** — it wins or ties
for best in 8 of 12 function/dim cells. Its tunable power exponent (between
`exp`'s `|d|¹` and `gauss`'s `|d|²`) lets it adapt the smoothness to the function.

**2. The right regression trend depends on the landscape:**
- *Smooth / polynomial-like* (sphere, low-dim rosenbrock) → **quadratic** wins.
  Sphere is literally a quadratic, so a quadratic trend fits it *exactly*
  (NRMSE = 0.0000) for every kernel — a nice sanity check.
- *Multimodal / high-dim* (rastrigin, ackley at d=10) → **constant** wins: the
  polynomial trend mis-extrapolates, so it's better to let the kernel carry the
  model. `linear` is frequently the *worst* — it adds parameters without
  capturing curvature.

**3. Difficulty scales as expected.** Best achievable NRMSE by function:
sphere ≈ 0 (trivial for a quadratic trend), rosenbrock 0.07 → 0.57 (d2→d10),
rastrigin 0.68 → 0.85, ackley 0.81 → 0.77. The multimodal functions stay near
1.0 for most configs — a Kriging surrogate from a modest uniform sample simply
can't resolve their high-frequency structure.

> Numbers are from one fixed-seed run; treat them as a ranking, not gospel.
> Re-run to explore other samples/dimensions.
