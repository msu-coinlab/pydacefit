# Surrogate benchmark

Which DACE **regression trend × correlation kernel × theta-mode** best surrogates
classical optimization test functions, across dimensions — and is there a single
configuration that is good *across the whole board*?

```bash
pyclawd python benchmark/benchmark.py                      # 11 funcs, d=2,5,10, full Friedman rank test
pyclawd python benchmark/benchmark.py --quick              # fast preset
pyclawd python benchmark/benchmark.py --dims 2 8 --ard off --brief
```

Each config fits DACE on a uniform sample (`n_train = 12·d`) and is scored on a
held-out test set by **NRMSE** = RMSE / std(test): `0` = perfect surrogate, `~1` =
no better than predicting the mean. Train/test data are fixed per problem so the
comparison is fair. Configs are ranked across all problems with a **Friedman rank
test** (re-implemented in numpy/stdlib; matches `scipy.stats.friedmanchisquare`),
with marginal rank tests by kernel, regression, and theta-mode.

Functions: `sphere, rosenbrock, rastrigin, ackley, griewank, zakharov, schwefel,
ellipsoid, levy, styblinski_tang, dixon_price` (the last seven re-implemented, not
imported from pymoo). 11 functions × dims {2,5,10} = 33 problems.

## Recommendation

**Rational Quadratic kernel (α ≈ 0.25) + constant regression + isotropic theta.**
It is the best *all-rounder*: lowest mean Friedman rank on both the 7- and
11-function suites, lowest mean NRMSE, and a worst case ≤ 0.99 (it never does worse
than predicting the mean on any problem). RQ is an infinite scale-mixture of
Gaussians, so it models smooth and rough structure at once; heavy tails (small α)
add robustness. It ships as `RationalQuadratic(alpha=...)`; among the fixed-exponent
kernels `Gaussian`/`GeneralizedExponential` are the closest (`expg` is the best of those).

```
config       mean_rank  mean_NRMSE   worst   avg gap vs per-problem oracle
rq (α=0.25)       2.15      0.4690   0.9906      +37.5%   ← best all-rounder
expg              2.71      0.5312   1.0001     +141.7%   (best built-in)
matern52          3.12      0.5132   0.9997     +263.7%
rq (α=0.25) ARD   3.15      0.5196   0.9626     +201.6%
gauss             3.86      0.5439   1.0108      +63.2%
Friedman p = 3.3e-4
```

## Learnings

1. **The regression trend is the biggest lever — and the most dangerous.**
   Quadratic wins on smooth/bowl functions (sphere, ellipsoid, griewank, zakharov;
   sphere is a literal quadratic → NRMSE ≈ 0) but is *actively harmful* on deceptive
   ones — quadratic on Schwefel blows up to NRMSE 1.55, **worse than the mean**.
   **Constant regression never hurts**, so it is the right choice for one robust model.

2. **α (kernel tail / smoothness) is problem-specific — no universal value.** The
   per-problem-best α spans the whole grid: **α ≈ 4** for smooth/bowl functions,
   **α ≈ 0.1** for rough/multimodal ones. Fixing α = 0.25 costs ~37 % vs the
   per-problem oracle (still the smallest gap of any fixed kernel).

3. **MLE hyperparameter tuning overfits on small samples — twice over.**
   - Tuning α *inside* DACE (boxmin = max-likelihood) is **worse** than fixing it.
   - **ARD loses to isotropic — even on the genuinely anisotropic `ellipsoid`**
     (ARD ~8× worse at d=10). It is not symmetry; it is that DACE fits its
     per-dimension length-scales by MLE on ~120 points, and the extra parameters
     overfit. pydacefit has **no cross-validation** (`fit.py` minimises
     `σ²·det(R)^(1/n)`), so there is no regularised selector to lean on.

4. **CV would help but costs 5–24×.** Selecting α by held-out error beats both fixed
   and MLE-tuned α, but each grid point re-runs the whole MLE search. For ~0.7 %
   accuracy over fixed α = 0.25 it is rarely worth it.

5. **Multimodal functions are intrinsically hard** to surrogate from a sparse
   uniform sample — rastrigin/ackley/schwefel/levy stay near NRMSE 0.8–1.0 for
   every config; more samples, not a better kernel, is what would move them.

**Practical takeaway:** default to RQ(α≈0.25) + constant + isotropic. If you know
the landscape is smooth push α up (→4), if rough push it down (→0.1). Skip ARD and
in-model α tuning on small samples; spend the budget on more training points instead.

> Numbers are from one fixed-seed run; treat them as a ranking, not gospel.
