"""Tests for predict(mse=True, grad=True).mse_grad: analytic variance gradient vs FD."""

import numpy as np
import pytest

from pydacefit.corr import Cubic, Exponential, Gaussian, Matern, RationalQuadratic
from pydacefit.dace import DACE
from pydacefit.regr import ConstantRegression, LinearRegression, QuadraticRegression


def _fit(regr, corr, d, seed=0, n=20):
    rng = np.random.default_rng(seed)
    X = rng.random((n, d))
    y = np.sum(np.sin(X * 3.0), axis=1)
    model = DACE(regr=regr, corr=corr, theta=0.5 * np.ones(d), thetaL=0.05 * np.ones(d), thetaU=10.0 * np.ones(d))
    model.fit(X, y)
    return model, X


def _fd_mse_grad(model, x, eps=1e-6):
    # central finite difference of the predictive variance w.r.t. the query point
    x = np.atleast_2d(x).astype(float)
    g = np.zeros(x.shape[1])
    for k in range(x.shape[1]):
        xp, xm = x.copy(), x.copy()
        xp[0, k] += eps
        xm[0, k] -= eps
        g[k] = (model.predict(xp, mse=True).mse[0, 0] - model.predict(xm, mse=True).mse[0, 0]) / (2 * eps)
    return g


CASES = [
    ("gauss/const", ConstantRegression(), Gaussian(), 2),
    ("gauss/linear", LinearRegression(), Gaussian(), 3),
    ("gauss/quad", QuadraticRegression(), Gaussian(), 2),
    ("exp/const", ConstantRegression(), Exponential(), 2),
    ("cubic/const", ConstantRegression(), Cubic(), 2),
    ("matern/linear", LinearRegression(), Matern(), 2),
    ("rq/const", ConstantRegression(), RationalQuadratic(alpha=1.0), 2),
]


@pytest.mark.parametrize("name,regr,corr,d", CASES, ids=[c[0] for c in CASES])
def test_mse_grad_matches_finite_difference(name, regr, corr, d):
    # the analytic variance gradient must match central finite differences of the MSE.
    # atol covers points where the gradient genuinely vanishes (FD relative error there
    # is meaningless); rtol covers the rest. The FD truncation floor is ~1e-7.
    # deterministic per-case seed (str hashing is randomized per process, so a failure
    # must stay reproducible across runs)
    model, X = _fit(regr, corr, d, seed=sum(ord(c) for c in name))
    rng = np.random.default_rng(1)
    queries = [rng.random((1, d)) for _ in range(4)] + [X[0:1] + 1e-3]  # last: near a train pt (sigma->0)
    for q in queries:
        p = model.predict(q, mse=True, grad=True)
        analytic = p.mse_grad[0]
        finite = _fd_mse_grad(model, q)
        assert np.allclose(analytic, finite, rtol=1e-4, atol=1e-6), (name, q, analytic, finite)


def test_mse_is_clamped_non_negative_at_training_points():
    # the predictive variance vanishes at training points and can round slightly negative;
    # predict clamps it at 0 so sqrt(mse) (std / EI) never returns NaN.
    model, X = _fit(ConstantRegression(), Gaussian(), 2, seed=3, n=25)
    mse = model.predict(X, mse=True).mse
    assert np.all(mse >= 0.0)
    assert np.all(np.isfinite(np.sqrt(mse)))


def test_mse_grad_present_only_with_both_flags():
    # mse_grad reuses the variance and mean-gradient terms, so it appears only when both
    # mse=True and grad=True; otherwise it is None.
    model, _ = _fit(ConstantRegression(), Gaussian(), 2)
    q = np.array([[0.3, 0.6]])
    assert model.predict(q).mse_grad is None
    assert model.predict(q, mse=True).mse_grad is None
    assert model.predict(q, grad=True).mse_grad is None
    p = model.predict(q, mse=True, grad=True)
    assert p.mse_grad is not None
    assert p.mse_grad.shape == q.shape  # (m, d)


def test_mse_grad_supports_std_gradient_for_ei():
    # the headline use case: grad(std) = mse_grad / (2 sqrt(mse)). Validate it against a
    # direct finite difference of std = sqrt(mse) at a generic (non-degenerate) point.
    model, _ = _fit(LinearRegression(), Gaussian(), 2, seed=5)
    q = np.array([[0.42, 0.18]])
    p = model.predict(q, mse=True, grad=True)
    std_grad = p.mse_grad[0] / (2.0 * np.sqrt(p.mse[0, 0]))

    eps = 1e-6
    fd = np.zeros(2)
    for k in range(2):
        qp, qm = q.copy(), q.copy()
        qp[0, k] += eps
        qm[0, k] -= eps
        fd[k] = (np.sqrt(model.predict(qp, mse=True).mse[0, 0]) - np.sqrt(model.predict(qm, mse=True).mse[0, 0])) / (
            2 * eps
        )
    assert np.allclose(std_grad, fd, rtol=1e-4, atol=1e-6)
