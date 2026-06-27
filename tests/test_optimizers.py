"""Tests for theta optimizers: analytic gradient correctness, L-BFGS-B, list-theta fix."""

import numpy as np

from pydacefit.corr import corr_gauss, corr_gauss_theta_grad
from pydacefit.dace import DACE
from pydacefit.fit import fit
from pydacefit.optimizers import LBFGS, objective_gradient, theta_grad_func
from pydacefit.regr import regr_constant


def _standardized(seed, d, n=18):
    # build standardized (nX, nY) the same way DACE.fit does internally
    rng = np.random.default_rng(seed)
    X = rng.random((n, d))
    y = np.sum(np.sin(X * 3.0), axis=1)[:, None]
    nX = (X - X.mean(0)) / X.std(0, ddof=1)
    nY = (y - y.mean(0)) / y.std(0, ddof=1)
    return nX, nY


def _obj(nX, nY, theta):
    return fit(nX, nY, regr_constant, corr_gauss, theta)["f"]


def _fd_grad(nX, nY, theta, eps=1e-6):
    # central finite difference of the objective w.r.t. each theta component
    theta = np.atleast_1d(np.array(theta, dtype=float))
    g = np.zeros_like(theta)
    for k in range(len(theta)):
        tp, tm = theta.copy(), theta.copy()
        tp[k] += eps
        tm[k] -= eps
        g[k] = (_obj(nX, nY, tp) - _obj(nX, nY, tm)) / (2 * eps)
    return g


# --- the core correctness check: analytic gradient == finite differences ---


def test_gauss_isotropic_gradient_matches_finite_difference():
    # multi-dim input with a single (length-1) theta -> exercises the sum-over-dims
    # isotropic branch of corr_gauss_theta_grad.
    nX, nY = _standardized(0, 3)
    theta = np.array([0.7])
    model = fit(nX, nY, regr_constant, corr_gauss, theta)

    analytic = objective_gradient(nX, model, theta, corr_gauss_theta_grad)
    finite = _fd_grad(nX, nY, theta)

    assert analytic.shape == (1,)
    assert np.allclose(analytic, finite, rtol=1e-5, atol=1e-7)


def test_gauss_ard_gradient_matches_finite_difference():
    # vector theta (one per input dim) -> per-dim ARD branch.
    nX, nY = _standardized(1, 3)
    theta = np.array([0.5, 1.3, 0.9])
    model = fit(nX, nY, regr_constant, corr_gauss, theta)

    analytic = objective_gradient(nX, model, theta, corr_gauss_theta_grad)
    finite = _fd_grad(nX, nY, theta)

    assert analytic.shape == (3,)
    assert np.allclose(analytic, finite, rtol=1e-5, atol=1e-7)


def test_gradient_matches_fd_across_several_thetas():
    # robustness: the match must hold away from any one lucky point.
    nX, nY = _standardized(4, 2)
    for theta in (np.array([0.2, 0.2]), np.array([1.0, 3.0]), np.array([5.0, 0.4])):
        model = fit(nX, nY, regr_constant, corr_gauss, theta)
        analytic = objective_gradient(nX, model, theta, corr_gauss_theta_grad)
        finite = _fd_grad(nX, nY, theta)
        assert np.allclose(analytic, finite, rtol=1e-4, atol=1e-6), theta


# --- the lookup wiring ---


def test_theta_grad_func_resolves_gauss_and_misses_unknown():
    assert theta_grad_func(corr_gauss) is corr_gauss_theta_grad

    def corr_made_up(D, theta):
        return np.ones(D.shape[0])

    assert theta_grad_func(corr_made_up) is None  # falls back to numeric gradient


# --- end-to-end L-BFGS-B with the analytic gradient ---


def test_lbfgs_improves_objective_and_converges_interior():
    rng = np.random.default_rng(2)
    X = rng.random((25, 2))
    y = np.sum(np.sin(X * 3.0), axis=1)

    lo, up = 1e-4, 50.0
    # tight tolerance here so the gradient-at-optimum assertion is meaningful
    model = DACE(
        regr=regr_constant,
        corr=corr_gauss,
        theta=np.array([1.0, 1.0]),
        thetaL=[lo, lo],
        thetaU=[up, up],
        optimizer=LBFGS(gtol=1e-8, ftol=1e-12),
    )
    model.fit(X, y)

    # the optimum must not be worse than the start point
    start_obj = fit(model.model["nX"], model.model["nY"], regr_constant, corr_gauss, np.array([1.0, 1.0]))["f"]
    assert model.model["f"] <= start_obj + 1e-9

    # for theta components strictly inside the bounds, the gradient must be ~0
    theta = np.ravel(model.model["theta"])
    grad = objective_gradient(model.model["nX"], model.model, theta, corr_gauss_theta_grad)
    interior = (theta > lo * 1.01) & (theta < up * 0.99)
    assert np.all(np.abs(grad[interior]) < 1e-3)
    assert np.all(np.isfinite(model.predict(rng.random((5, 2)))))


def test_relaxed_lbfgs_saves_evaluations_and_stays_close():
    # the whole point of the relaxed default: fewer objective evaluations than a
    # tight run, while still landing on essentially the same optimum.
    import pydacefit.optimizers.lbfgs as lb

    rng = np.random.default_rng(3)
    X = rng.random((22, 2))
    y = np.sum(np.sin(X * 3.0), axis=1)

    def _build(optimizer):
        return DACE(
            regr=regr_constant,
            corr=corr_gauss,
            theta=np.array([1.0, 1.0]),
            thetaL=[1e-4, 1e-4],
            thetaU=[50.0, 50.0],
            optimizer=optimizer,
        )

    calls = [0]
    real_fit = lb.fit

    def counting(*a, **k):
        calls[0] += 1
        return real_fit(*a, **k)

    lb.fit = counting
    try:
        calls[0] = 0
        relaxed = _build(LBFGS(gtol=1e-2))
        relaxed.fit(X, y)
        relaxed_evals = calls[0]

        calls[0] = 0
        tight = _build(LBFGS(gtol=1e-9, ftol=1e-14))
        tight.fit(X, y)
        tight_evals = calls[0]
    finally:
        lb.fit = real_fit

    assert relaxed_evals < tight_evals  # relaxing the tolerance really saves work
    assert relaxed.model["f"] <= tight.model["f"] * 1.02 + 1e-9  # still a good optimum


def test_lbfgs_matches_boxmin_when_optimum_is_unique():
    # on a well-conditioned 1d problem both optimizers find the same single optimum.
    rng = np.random.default_rng(11)
    X = rng.random((20, 1))
    y = np.sum(np.sin(X * 2 * np.pi), axis=1)

    box = DACE(regr=regr_constant, corr=corr_gauss, theta=1.0, thetaL=1e-4, thetaU=100.0)
    box.fit(X, y)
    lb = DACE(regr=regr_constant, corr=corr_gauss, theta=1.0, thetaL=1e-4, thetaU=100.0, optimizer=LBFGS())
    lb.fit(X, y)

    xt = np.linspace(0, 1, 40)[:, None]
    assert np.allclose(box.predict(xt), lb.predict(xt), atol=1e-2)


def test_lbfgs_restarts_escape_a_bad_starting_basin():
    # the DACE likelihood is multi-modal: from a poor start theta (here the lower
    # bound, a flat plateau where the gradient vanishes) a single L-BFGS-B run stays
    # stuck. Random restarts must escape it and reach the good optimum.
    rng = np.random.default_rng(3)
    X = rng.random((30, 1))
    y = np.sin(X[:, 0] * 12.0)

    bad_start = 1e-4  # the lower bound -> single start cannot move off the plateau
    single = DACE(regr=regr_constant, corr=corr_gauss, theta=bad_start, thetaL=1e-4, thetaU=100.0, optimizer=LBFGS())
    single.fit(X, y)

    restarts = DACE(
        regr=regr_constant, corr=corr_gauss, theta=bad_start, thetaL=1e-4, thetaU=100.0, optimizer=LBFGS(n_restarts=12)
    )
    restarts.fit(X, y)

    # restarts find a far better objective and recover the (near-interpolating) fit
    assert restarts.model["f"] < single.model["f"] - 1.0
    assert restarts.model["f"] < 1e-3
    assert restarts.model["theta"].ravel()[0] > bad_start  # actually moved off the plateau


def test_lbfgs_first_start_is_the_configured_theta():
    # n_restarts must not change the warm-start contract: the first start is always
    # the configured theta, so a feasible warm start can only be improved upon.
    rng = np.random.default_rng(5)
    X = rng.random((20, 1))
    y = np.sum(np.sin(X * 2 * np.pi), axis=1)

    plain = DACE(regr=regr_constant, corr=corr_gauss, theta=1.0, thetaL=1e-4, thetaU=100.0, optimizer=LBFGS())
    plain.fit(X, y)
    multi = DACE(
        regr=regr_constant, corr=corr_gauss, theta=1.0, thetaL=1e-4, thetaU=100.0, optimizer=LBFGS(n_restarts=5)
    )
    multi.fit(X, y)

    # multi-start keeps the best, so it is never worse than the single warm start
    assert multi.model["f"] <= plain.model["f"] + 1e-9


# --- pre-existing bug fix: a Python-list theta (ARD) no longer crashes boxmin ---


def test_list_theta_ard_fits_via_boxmin():
    # regression: a list theta used to reach corr_gauss as a list and crash with
    # "bad operand type for unary -: 'list'"; __init__ now coerces it to an array.
    rng = np.random.default_rng(0)
    X = rng.random((20, 2))
    y = np.sum(np.sin(X * 3.0), axis=1)

    model = DACE(regr=regr_constant, corr=corr_gauss, theta=[1.0, 1.0], thetaL=[1e-4, 1e-4], thetaU=[20.0, 20.0])
    model.fit(X, y)
    assert np.all(np.isfinite(model.predict(rng.random((4, 2)))))
