"""Benchmark: which DACE regression x kernel best surrogates classical test functions.

Trains a DACE model (with ARD theta-optimization) on a uniform sample of each
function, then scores surrogate accuracy on a held-out test set via NRMSE — the
RMSE normalized by the test-set standard deviation, so it is comparable across
functions and dimensions (~0 = perfect surrogate, ~1 = no better than the mean).

Run:  pyclawd python benchmark/surrogate_benchmark.py [--dims 2 5 10] [--quick]
"""

import argparse
import time
import warnings

import numpy as np

from pydacefit.corr import (
    corr_cubic,
    corr_exp,
    corr_expg,
    corr_gauss,
    corr_lin,
    corr_spherical,
    corr_spline,
)
from pydacefit.dace import DACE
from pydacefit.regr import regr_constant, regr_linear, regr_quadratic

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


FUNCS = {
    "sphere": (sphere, -5.0, 5.0),
    "rosenbrock": (rosenbrock, -2.0, 2.0),
    "rastrigin": (rastrigin, -5.12, 5.12),
    "ackley": (ackley, -32.768, 32.768),
    "griewank": (griewank, -600.0, 600.0),
    "zakharov": (zakharov, -10.0, 10.0),
    "schwefel": (schwefel, -500.0, 500.0),
}

REGR = {"constant": regr_constant, "linear": regr_linear, "quadratic": regr_quadratic}

KERNELS = {
    "gauss": corr_gauss,
    "exp": corr_exp,
    "cubic": corr_cubic,
    "spline": corr_spline,
    "spherical": corr_spherical,
    "lin": corr_lin,
    "expg": corr_expg,
}


def _theta_config(kernel_name, d):
    # expg's theta is (length-scale, power) and is isotropic; the rest use a
    # per-dimension (ARD) length-scale vector. All bounds drive the boxmin search.
    if kernel_name == "expg":
        return dict(theta=np.array([1.0, 2.0]), thetaL=np.array([0.01, 1.0]), thetaU=np.array([20.0, 3.0]))
    return dict(theta=np.full(d, 1.0), thetaL=np.full(d, 0.01), thetaU=np.full(d, 20.0))


def _sample(rng, n, d, lo, hi):
    return lo + (hi - lo) * rng.random((n, d))


def _nrmse(pred, true):
    rmse = float(np.sqrt(np.mean((pred.ravel() - true) ** 2)))
    spread = float(np.std(true))
    return rmse / spread if spread > 0 else rmse


def evaluate(func, lo, hi, d, n_train, n_test):
    # identical train/test data for every config -> a fair comparison
    Xtr = _sample(np.random.RandomState(42), n_train, d, lo, hi)
    Xte = _sample(np.random.RandomState(7), n_test, d, lo, hi)
    ytr, yte = func(Xtr), func(Xte)

    results = {}
    for rname, regr in REGR.items():
        for kname, kernel in KERNELS.items():
            try:
                model = DACE(regr=regr, corr=kernel, **_theta_config(kname, d))
                model.fit(Xtr, ytr)
                pred = model.predict(Xte)
                score = _nrmse(pred, yte)
                results[(rname, kname)] = score if np.isfinite(score) else None
            except Exception:
                results[(rname, kname)] = None
    return results


def _print_block(fname, d, n_train, n_test, results):
    print(f"\n### {fname}  d={d}   (n_train={n_train}, n_test={n_test}) — NRMSE, lower is better ###")
    best = min((s, rk) for rk, s in results.items() if s is not None)
    best_score, best_rk = best
    header = "kernel".ljust(11) + "".join(r.rjust(11) for r in REGR)
    print(header)
    for kname in KERNELS:
        row = kname.ljust(11)
        for rname in REGR:
            s = results[(rname, kname)]
            cell = "  fail" if s is None else f"{s:.4f}"
            if (rname, kname) == best_rk:
                cell += "*"
            row += cell.rjust(11)
        print(row)
    print(f"  best: {best_rk[0]} + {best_rk[1]}   NRMSE={best_score:.4f}")
    return best_rk


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dims", type=int, nargs="+", default=[2, 5, 10])
    ap.add_argument("--funcs", nargs="+", default=list(FUNCS), choices=list(FUNCS))
    ap.add_argument("--train-per-dim", type=int, default=12, help="n_train = train_per_dim * d")
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--quick", action="store_true", help="small/fast preset (dims 2,5; 2 funcs)")
    args = ap.parse_args()

    dims = [2, 5] if args.quick else args.dims
    funcs = ["rosenbrock", "rastrigin"] if args.quick else args.funcs

    t0 = time.perf_counter()
    wins = {}
    for fname in funcs:
        func, lo, hi = FUNCS[fname]
        for d in dims:
            n_train = args.train_per_dim * d
            res = evaluate(func, lo, hi, d, n_train, args.n_test)
            best_rk = _print_block(fname, d, n_train, args.n_test, res)
            wins[best_rk] = wins.get(best_rk, 0) + 1

    print(f"\n=== winners tally (best config per function/dim), {time.perf_counter() - t0:.1f}s ===")
    for rk, n in sorted(wins.items(), key=lambda kv: -kv[1]):
        print(f"  {n:2d}x   {rk[0]} + {rk[1]}")


if __name__ == "__main__":
    main()
