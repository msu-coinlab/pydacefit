"""Multi-output (q>1) coverage: shared-hyperparameter GP gradient + refit round-trip.

The per-output sum in the objective (``sum_j sigma2_j``) and its envelope-theorem
gradient term are where a multi-output bug would hide, so the analytic
``objective_gradient`` is checked against finite differences with a matrix ``Y``;
``refit`` is exercised end-to-end with matrix targets.
"""

import numpy as np

from pydacefit.corr import Gaussian
from pydacefit.dace import DACE
from pydacefit.fit import fit
from pydacefit.optimizers import LBFGS, Boxmin, objective_gradient
from pydacefit.regr import ConstantRegression

GAUSS = Gaussian()
CONSTANT = ConstantRegression()


def _standardized_multi(seed, d=2, q=2, n=20):
    rng = np.random.default_rng(seed)
    X = rng.random((n, d))
    cols = [np.sin(X[:, 0] * 3.0 + j) + np.cos(X[:, 1] * 2.0 - j) for j in range(q)]
    Y = np.column_stack(cols)
    nX = (X - X.mean(0)) / X.std(0, ddof=1)
    nY = (Y - Y.mean(0)) / Y.std(0, ddof=1)
    return nX, nY


def _obj(nX, nY, theta):
    return fit(nX, nY, CONSTANT, GAUSS, theta)["f"]


def _fd_grad(nX, nY, theta, eps=1e-6):
    theta = np.atleast_1d(np.array(theta, dtype=float))
    g = np.zeros_like(theta)
    for k in range(len(theta)):
        tp, tm = theta.copy(), theta.copy()
        tp[k] += eps
        tm[k] -= eps
        g[k] = (_obj(nX, nY, tp) - _obj(nX, nY, tm)) / (2 * eps)
    return g


def test_objective_gradient_matches_fd_with_matrix_Y():
    # the analytic gradient sums tr/quad terms over a single shared factorization but
    # the objective sums sigma2 over all q outputs -- FD pins that the per-output sum
    # and the envelope-theorem cancellation are handled correctly for q>1.
    nX, nY = _standardized_multi(0, d=2, q=3)
    for theta in (np.array([0.6, 0.6]), np.array([1.2, 0.4]), np.array([3.0, 2.0])):
        model = fit(nX, nY, CONSTANT, GAUSS, theta)
        analytic = objective_gradient(nX, model, theta, GAUSS.theta_grad)
        finite = _fd_grad(nX, nY, theta)
        assert analytic.shape == (2,)
        assert np.allclose(analytic, finite, rtol=1e-4, atol=1e-6), theta


def test_refit_roundtrip_with_matrix_Y():
    # refit on matrix targets must append the new points, keep the (m, q) prediction
    # shape, and land essentially on the cold fit of the combined data. It is not
    # bit-identical: the shared-theta multi-output likelihood is flatter, so Boxmin's
    # pattern search settles in a marginally different basin from a warm vs cold start
    # (a ~3e-4 prediction difference here) -- the destination matches to a loose tol.
    rng = np.random.default_rng(1)
    X0 = rng.random((16, 2))
    Xn = rng.random((8, 2))
    X_all = np.vstack([X0, Xn])

    def _Y(X):
        return np.column_stack([np.sum(np.sin(X * 3.0), axis=1), np.sum(np.cos(X * 2.0), axis=1)])

    def _model():
        return DACE(regr=CONSTANT, corr=GAUSS, theta=1.0, thetaL=1e-4, thetaU=50.0)

    cold = _model()
    cold.fit(X_all, _Y(X_all))

    warm = _model()
    warm.fit(X0, _Y(X0))
    # opt into the cold fit's semantics (MLE + same optimizer); the refit defaults
    # (warm LBFGS + validation=True) optimize a different objective.
    warm.refit(Xn, _Y(Xn), optimizer=Boxmin(), validation=False)

    assert warm.model["X"].shape[0] == 24
    xt = rng.random((10, 2))
    assert warm.predict(xt).y.shape == (10, 2)
    assert np.all(np.isfinite(warm.predict(xt).y))
    assert np.allclose(cold.predict(xt).y, warm.predict(xt).y, atol=5e-3)


def test_lbfgs_objective_gradient_matrix_Y_end_to_end():
    # the analytic-gradient LBFGS path must drive a matrix-Y fit to a valid optimum.
    rng = np.random.default_rng(2)
    X = rng.random((22, 2))
    Y = np.column_stack([np.sum(np.sin(X * 3.0), axis=1), np.sum(np.cos(X * 2.0), axis=1)])

    model = DACE(
        regr=CONSTANT,
        corr=GAUSS,
        theta=np.array([1.0, 1.0]),
        thetaL=[1e-4, 1e-4],
        thetaU=[50.0, 50.0],
        optimizer=LBFGS(options={"gtol": 1e-8, "ftol": 1e-12}),
    )
    model.fit(X, Y)

    theta = np.ravel(model.model["theta"])
    grad = objective_gradient(model.model["nX"], model.model, theta, GAUSS.theta_grad)
    interior = (theta > 1e-4 * 1.01) & (theta < 50.0 * 0.99)
    assert np.all(np.abs(grad[interior]) < 1e-3)
    assert model.predict(rng.random((5, 2))).y.shape == (5, 2)
