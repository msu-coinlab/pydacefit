"""Behavior tests for DACE.refit — append new points and re-fit, warm-started."""

import numpy as np
import pytest

from pydacefit.corr import corr_gauss
from pydacefit.dace import DACE
from pydacefit.optimizers import LBFGS, Boxmin, Fixed
from pydacefit.regr import regr_constant


def _fun(X):
    return np.sum(np.sin(X * 2 * np.pi), axis=1)


def _model(theta=1.0, thetaL=1e-5, thetaU=100.0, optimizer=None):
    return DACE(regr=regr_constant, corr=corr_gauss, theta=theta, thetaL=thetaL, thetaU=thetaU, optimizer=optimizer)


def test_refit_before_fit_raises():
    rng = np.random.default_rng(0)
    X = rng.random((5, 1))
    with pytest.raises(Exception, match="requires a prior fit"):
        _model().refit(X, _fun(X))


def test_refit_appends_and_matches_cold_fit_on_combined_data():
    # refit takes only the new points; the result must equal a cold fit on the
    # full combined set (warm starting changes the path, not the destination).
    rng = np.random.default_rng(0)
    X0 = rng.random((15, 1))
    X_new = rng.random((10, 1))
    X_all = np.vstack([X0, X_new])

    cold = _model()
    cold.fit(X_all, _fun(X_all))

    warm = _model()
    warm.fit(X0, _fun(X0))
    warm.refit(X_new, _fun(X_new))  # only the additions

    x_test = np.linspace(0, 1, 50)[:, None]
    assert np.allclose(cold.predict(x_test), warm.predict(x_test), atol=1e-5)


def test_refit_grows_the_stored_training_set():
    rng = np.random.default_rng(3)
    X0 = rng.random((12, 1))
    model = _model()
    model.fit(X0, _fun(X0))
    assert model.model["X"].shape[0] == 12

    X_new = rng.random((7, 1))
    model.refit(X_new, _fun(X_new))
    assert model.model["X"].shape[0] == 19
    assert model.model["nX"].shape[0] == 19  # the fit actually used all points


def test_refit_warm_starts_from_previous_theta():
    # the search seeds from self.theta; after fit() that should be the previous
    # optimized theta, not the original initial guess of 1.0.
    rng = np.random.default_rng(1)
    X0 = rng.random((12, 1))
    model = _model(theta=1.0)
    model.fit(X0, _fun(X0))
    optimized = model.model["theta"].copy()

    X_new = rng.random((6, 1))
    model.refit(X_new, _fun(X_new))
    assert np.allclose(model.theta, optimized)
    assert not np.allclose(model.theta, 1.0)


def test_refit_fixed_optimizer_freezes_theta_but_uses_new_data():
    # Fixed() must not move theta, yet must still incorporate the new points
    # (prediction changes because the data grew, not because theta did).
    rng = np.random.default_rng(7)
    X0 = rng.random((12, 1))
    model = _model()
    model.fit(X0, _fun(X0))
    theta_before = model.model["theta"].copy()

    x_test = np.linspace(0, 1, 30)[:, None]
    pred_before = model.predict(x_test)

    X_new = rng.random((6, 1))
    model.refit(X_new, _fun(X_new), optimizer=Fixed())

    assert np.allclose(model.model["theta"], theta_before)  # theta frozen
    assert model.model["X"].shape[0] == 18  # data still appended
    assert not np.allclose(pred_before, model.predict(x_test))  # fit changed


def test_refit_optimizer_override_is_per_call():
    # passing an optimizer to refit must not change the model's configured one.
    rng = np.random.default_rng(8)
    X0 = rng.random((10, 1))
    configured = Boxmin()
    model = _model(optimizer=configured)
    model.fit(X0, _fun(X0))

    model.refit(rng.random((4, 1)), _fun(rng.random((4, 1))), optimizer=Fixed())
    assert model.optimizer is configured  # restored after the override


def test_refit_with_lbfgs_override_produces_valid_fit():
    # a per-call lbfgs refine must append the data, keep theta within bounds, and
    # yield finite predictions. (It need not match boxmin's optimum exactly -- on
    # a flat likelihood the two local searches can settle on different theta.)
    rng = np.random.default_rng(5)
    X0 = rng.random((15, 1))
    X_new = rng.random((8, 1))

    model = _model()
    model.fit(X0, _fun(X0))
    model.refit(X_new, _fun(X_new), optimizer=LBFGS())

    assert model.model["X"].shape[0] == 23
    theta = np.ravel(model.model["theta"])
    assert np.all((theta >= 1e-5) & (theta <= 100.0))
    assert np.all(np.isfinite(model.predict(np.linspace(0, 1, 40)[:, None])))


def test_lbfgs_as_configured_optimizer():
    # LBFGS() works as the model-level configured optimizer too.
    rng = np.random.default_rng(6)
    X0 = rng.random((12, 1))
    model = _model(optimizer=LBFGS())
    model.fit(X0, _fun(X0))
    assert model.itpar is None  # lbfgs leaves no boxmin trajectory
    pred = model.predict(np.linspace(0, 1, 10)[:, None])
    assert np.all(np.isfinite(pred))


def test_refit_without_optimization_reuses_theta():
    # no bounds -> no hyperparameter search; refit keeps the last theta.
    rng = np.random.default_rng(2)
    X0 = rng.random((10, 1))
    model = DACE(regr=regr_constant, corr=corr_gauss, theta=2.0, thetaL=None, thetaU=None)
    model.fit(X0, _fun(X0))

    X_new = rng.random((5, 1))
    model.refit(X_new, _fun(X_new))
    assert np.allclose(model.model["theta"], 2.0)
