"""Benchmark: which DACE regression x kernel x theta-mode best surrogates test functions.

Trains a DACE model on a uniform sample of each classical function, then scores
surrogate accuracy on a held-out test set via NRMSE — the RMSE normalized by the
test-set standard deviation, so it is comparable across functions and dimensions
(~0 = perfect surrogate, ~1 = no better than the mean).

theta-mode is a comparison axis: "ard" optimizes a per-dimension length-scale
vector (Automatic Relevance Determination), "iso" optimizes a single shared
scalar length-scale. Configs are ranked across all problems with a Friedman rank
test, plus marginal rank tests by kernel, by regression, and by theta-mode.

Run:  pyclawd python benchmark/benchmark.py [--dims 2 5 10] [--ard both|on|off] [--brief]
"""

import argparse
import math
import time
import warnings

import numpy as np

from pydacefit.corr import (
    Cubic,
    Exponential,
    Gaussian,
    GeneralizedExponential,
    Linear,
    Spherical,
    Spline,
)
from pydacefit.dace import DACE
from pydacefit.regr import ConstantRegression, LinearRegression, QuadraticRegression

warnings.filterwarnings("ignore")  # silence ill-conditioned RuntimeWarnings during search


# --- classical test functions (minimization); each paired with its [lo, hi] box ---


def sphere(X):
    return np.sum(X**2, axis=1)


def rosenbrock(X):
    return np.sum(100.0 * (X[:, 1:] - X[:, :-1] ** 2) ** 2 + (1.0 - X[:, :-1]) ** 2, axis=1)


def rastrigin(X):
    d = X.shape[1]
    return 10.0 * d + np.sum(X**2 - 10.0 * np.cos(2.0 * np.pi * X), axis=1)


def ackley(X):
    d = X.shape[1]
    s1 = np.sum(X**2, axis=1)
    s2 = np.sum(np.cos(2.0 * np.pi * X), axis=1)
    return -20.0 * np.exp(-0.2 * np.sqrt(s1 / d)) - np.exp(s2 / d) + 20.0 + np.e


# the next three are re-implemented from pymoo's single-objective suite


def griewank(X):
    i = np.arange(1, X.shape[1] + 1)
    return 1.0 + np.sum(X**2, axis=1) / 4000.0 - np.prod(np.cos(X / np.sqrt(i)), axis=1)


def zakharov(X):
    i = np.arange(1, X.shape[1] + 1)
    a = 0.5 * np.sum(i * X, axis=1)
    return np.sum(X**2, axis=1) + a**2 + a**4


def schwefel(X):
    d = X.shape[1]
    return 418.9829 * d - np.sum(X * np.sin(np.sqrt(np.abs(X))), axis=1)


def ellipsoid(X):
    # anisotropic smooth bowl: dimension i has weight i -> the case where ARD matters
    w = np.arange(1, X.shape[1] + 1)
    return np.sum(w * X**2, axis=1)


def levy(X):
    w = 1.0 + (X - 1.0) / 4.0
    term1 = np.sin(np.pi * w[:, 0]) ** 2
    mid = np.sum((w[:, :-1] - 1.0) ** 2 * (1.0 + 10.0 * np.sin(np.pi * w[:, :-1] + 1.0) ** 2), axis=1)
    term3 = (w[:, -1] - 1.0) ** 2 * (1.0 + np.sin(2.0 * np.pi * w[:, -1]) ** 2)
    return term1 + mid + term3


def styblinski_tang(X):
    return 0.5 * np.sum(X**4 - 16.0 * X**2 + 5.0 * X, axis=1)


def dixon_price(X):
    i = np.arange(2, X.shape[1] + 1)
    return (X[:, 0] - 1.0) ** 2 + np.sum(i * (2.0 * X[:, 1:] ** 2 - X[:, :-1]) ** 2, axis=1)


FUNCS = {
    "sphere": (sphere, -5.0, 5.0),
    "rosenbrock": (rosenbrock, -2.0, 2.0),
    "rastrigin": (rastrigin, -5.12, 5.12),
    "ackley": (ackley, -32.768, 32.768),
    "griewank": (griewank, -600.0, 600.0),
    "zakharov": (zakharov, -10.0, 10.0),
    "schwefel": (schwefel, -500.0, 500.0),
    "ellipsoid": (ellipsoid, -5.0, 5.0),
    "levy": (levy, -10.0, 10.0),
    "styblinski_tang": (styblinski_tang, -5.0, 5.0),
    "dixon_price": (dixon_price, -10.0, 10.0),
}

REGR = {"constant": ConstantRegression(), "linear": LinearRegression(), "quadratic": QuadraticRegression()}

KERNELS = {
    "gauss": Gaussian(),
    "exp": Exponential(),
    "cubic": Cubic(),
    "spline": Spline(),
    "spherical": Spherical(),
    "lin": Linear(),
    "expg": GeneralizedExponential(),
}


def _theta_config(kernel_name, d, mode):
    # expg's theta is (length-scale, power) and is inherently isotropic. Otherwise
    # "ard" optimizes a per-dimension length-scale vector, "iso" a single scalar.
    if kernel_name == "expg":
        return dict(theta=np.array([1.0, 2.0]), thetaL=np.array([0.01, 1.0]), thetaU=np.array([20.0, 3.0]))
    if mode == "ard":
        return dict(theta=np.full(d, 1.0), thetaL=np.full(d, 0.01), thetaU=np.full(d, 20.0))
    return dict(theta=1.0, thetaL=0.01, thetaU=20.0)


def configs_for(ard_mode):
    # config = (regr, kernel, theta_mode). expg appears once (iso only).
    modes = {"both": ["iso", "ard"], "on": ["ard"], "off": ["iso"]}[ard_mode]
    out = []
    for r in REGR:
        for k in KERNELS:
            if k == "expg":
                out.append((r, k, "iso"))
            else:
                out.extend((r, k, m) for m in modes)
    return out


def _sample(rng, n, d, lo, hi):
    return lo + (hi - lo) * rng.random((n, d))


def _nrmse(pred, true):
    rmse = float(np.sqrt(np.mean((pred.ravel() - true) ** 2)))
    spread = float(np.std(true))
    return rmse / spread if spread > 0 else rmse


def evaluate(func, lo, hi, d, n_train, n_test, configs):
    # identical train/test data for every config -> a fair comparison
    Xtr = _sample(np.random.RandomState(42), n_train, d, lo, hi)
    Xte = _sample(np.random.RandomState(7), n_test, d, lo, hi)
    ytr, yte = func(Xtr), func(Xte)

    results = {}
    for rname, kname, mode in configs:
        try:
            model = DACE(regr=REGR[rname], corr=KERNELS[kname], **_theta_config(kname, d, mode))
            model.fit(Xtr, ytr)
            score = _nrmse(model.predict(Xte).y, yte)
            results[(rname, kname, mode)] = score if np.isfinite(score) else None
        except Exception:
            results[(rname, kname, mode)] = None
    return results


def _print_block(fname, d, n_train, n_test, results):
    print(f"\n### {fname}  d={d}   (n_train={n_train}, n_test={n_test}) — NRMSE, lower is better ###")
    best_rk = min((rk for rk, s in results.items() if s is not None), key=lambda rk: results[rk])
    rows = []
    for r, k, m in results:
        if (k, m) not in rows:
            rows.append((k, m))
    print("kernel/mode".ljust(16) + "".join(r.rjust(11) for r in REGR))
    for k, m in rows:
        row = f"{k}/{m}".ljust(16)
        for r in REGR:
            s = results.get((r, k, m))
            cell = "  -" if (r, k, m) not in results else ("fail" if s is None else f"{s:.4f}")
            if (r, k, m) == best_rk:
                cell += "*"
            row += cell.rjust(11)
        print(row)
    print(f"  best: {best_rk[0]} + {best_rk[1]}/{best_rk[2]}   NRMSE={results[best_rk]:.4f}")


# --- rank statistics (Friedman test), re-implemented in numpy/stdlib ---


def _rankdata_avg(values):
    # 1-based ranks, ascending (rank 1 = smallest = best); ties share the average rank
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1)
    s = values[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j + 2) / 2.0  # mean of the 1-based tied ranks
        i = j + 1
    return ranks


def _chi2_sf(x, df):
    # survival function of the chi-square distribution = regularized upper incomplete
    # gamma Q(df/2, x/2), via series (small x) or continued fraction (large x).
    s, y = df / 2.0, x / 2.0
    if y <= 0:
        return 1.0
    if y < s + 1.0:  # lower series for P, then Q = 1 - P
        term = total = 1.0 / s
        a = s
        for _ in range(2000):
            a += 1.0
            term *= y / a
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        return 1.0 - total * math.exp(-y + s * math.log(y) - math.lgamma(s))
    tiny = 1e-300  # continued fraction for Q directly
    b, c, d = y + 1.0 - s, 1.0 / tiny, 1.0 / (y + 1.0 - s)
    h = d
    for i in range(1, 2000):
        an = -i * (i - s)
        b += 2.0
        d = an * d + b
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = b + an / (c if abs(c) > tiny else tiny)
        h *= d * c
        if abs(d * c - 1.0) < 1e-15:
            break
    return math.exp(-y + s * math.log(y) - math.lgamma(s)) * h


def _friedman(score_matrix):
    # score_matrix: (n_problems, k_treatments), lower=better. Returns ranking + test.
    rank_matrix = np.vstack([_rankdata_avg(row) for row in score_matrix])
    n, k = rank_matrix.shape
    rank_sums = rank_matrix.sum(axis=0)
    chi2 = 12.0 / (n * k * (k + 1)) * np.sum(rank_sums**2) - 3.0 * n * (k + 1)
    return rank_matrix.mean(axis=0), chi2, k - 1, _chi2_sf(chi2, k - 1)


def _matrix(all_results, configs):
    problems = list(all_results)
    mat = np.full((len(problems), len(configs)), np.inf)  # failures -> +inf rank last
    for pi, prob in enumerate(problems):
        for ci, cfg in enumerate(configs):
            v = all_results[prob].get(cfg)
            if v is not None and np.isfinite(v):
                mat[pi, ci] = v
    return mat, problems


def _marginal(label, groups, mat, configs):
    # per problem, score each group by its best member, then rank the groups
    score = np.full((mat.shape[0], len(groups)), np.inf)
    for gi, members in enumerate(groups.values()):
        cols = [configs.index(m) for m in members]
        score[:, gi] = mat[:, cols].min(axis=1)
    mrank, chi2, df, p = _friedman(score)
    print(f"\nmean rank by {label} (best-of-group per problem, lower=better):")
    for gi, name in sorted(enumerate(groups), key=lambda t: mrank[t[0]]):
        print(f"  {name:<12}{mrank[gi]:>6.2f}")
    print(f"  Friedman: chi2={chi2:.1f}, df={df}, p={p:.2e}")


def analyze(all_results, configs):
    mat, problems = _matrix(all_results, configs)
    mean_rank, chi2, df, p = _friedman(mat)
    finite = np.isfinite(mat)
    mean_err = np.array([mat[finite[:, c], c].mean() if finite[:, c].any() else math.nan for c in range(len(configs))])
    fails = (~finite).sum(axis=0)

    print(f"\n=== overall ranking across {len(problems)} problems (Friedman rank test) ===")
    print(f"{'#':>3}  {'config':<26}{'mean NRMSE':>12}{'mean rank':>11}{'fails':>7}")
    for pos, ci in enumerate(sorted(range(len(configs)), key=lambda c: mean_rank[c]), 1):
        r, k, m = configs[ci]
        print(f"{pos:>3}  {f'{r} + {k}/{m}':<26}{mean_err[ci]:>12.4f}{mean_rank[ci]:>11.2f}{fails[ci]:>7}")
    sig = "significant" if p < 0.05 else "not significant"
    print(f"\nFriedman chi2 = {chi2:.1f}, df = {df}, p = {p:.2e}  ->  {sig} at a=0.05")
    print(f"(k = {len(configs)} configs ranked across N = {len(problems)} problems)")

    _marginal("KERNEL", {k: [c for c in configs if c[1] == k] for k in KERNELS}, mat, configs)
    _marginal("REGRESSION", {r: [c for c in configs if c[0] == r] for r in REGR}, mat, configs)
    # iso vs ard only over kernels that support both (exclude expg for a fair match)
    both = {m: [c for c in configs if c[2] == m and c[1] != "expg"] for m in ("iso", "ard")}
    if all(both.values()):
        _marginal("THETA-MODE", both, mat, configs)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dims", type=int, nargs="+", default=[2, 5, 10])
    ap.add_argument("--funcs", nargs="+", default=list(FUNCS), choices=list(FUNCS))
    ap.add_argument("--ard", choices=["both", "on", "off"], default="both", help="theta-mode axis")
    ap.add_argument("--train-per-dim", type=int, default=12, help="n_train = train_per_dim * d")
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--quick", action="store_true", help="small/fast preset (dims 2,5; 2 funcs)")
    ap.add_argument("--brief", action="store_true", help="skip per-problem tables, show only the ranking")
    args = ap.parse_args()

    dims = [2, 5] if args.quick else args.dims
    funcs = ["rosenbrock", "rastrigin"] if args.quick else args.funcs
    configs = configs_for(args.ard)

    t0 = time.perf_counter()
    all_results = {}
    for fname in funcs:
        func, lo, hi = FUNCS[fname]
        for d in dims:
            n_train = args.train_per_dim * d
            res = evaluate(func, lo, hi, d, n_train, args.n_test, configs)
            if not args.brief:
                _print_block(fname, d, n_train, args.n_test, res)
            all_results[(fname, d)] = res

    analyze(all_results, configs)
    print(f"\n({len(all_results)} problems x {len(configs)} configs in {time.perf_counter() - t0:.1f}s)")


if __name__ == "__main__":
    main()
