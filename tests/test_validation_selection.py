"""Behavior tests for validation-set theta selection (Optimizer(validation=...))."""

import numpy as np
import pytest

from pydacefit.corr import corr_gauss
from pydacefit.dace import DACE
from pydacefit.optimizers import LBFGS, Boxmin
from pydacefit.regr import regr_constant


def _fun(X):
    return np.sum(np.sin(X * 2 * np.pi), axis=1)


def _model(optimizer=None):
    return DACE(regr=regr_constant, corr=corr_gauss, theta=1.0, thetaL=1e-5, thetaU=100.0, optimizer=optimizer)


def _rmse(a, b):
    a = a if a.ndim == 2 else a[:, None]
    b = b if b.ndim == 2 else b[:, None]
    return float(np.sqrt(np.mean(np.square(a - b))))


def test_validation_none_matches_default_boxmin():
    # the opt-out path must be byte-for-byte the historical MLE behavior: passing
    # validation=None (or no optimizer at all) selects theta by likelihood.
    rng = np.random.default_rng(0)
    X = rng.random((20, 1))
    y = _fun(X)

    default = _model()
    default.fit(X, y)
    explicit = _model(optimizer=Boxmin(validation=None))
    explicit.fit(X, y)

    assert np.allclose(default.model["theta"], explicit.model["theta"])
    xt = np.linspace(0, 1, 50)[:, None]
    assert np.allclose(default.predict(xt), explicit.predict(xt))


def test_val_error_equals_predict_rmse_in_original_space():
    # _val_error must measure exactly what predict() would, in original Y units.
    rng = np.random.default_rng(1)
    X = rng.random((18, 1))
    y = _fun(X)
    Xv = rng.random((9, 1))
    yv = _fun(Xv)

    m = _model()
    m.fit(X, y)

    internal = m._val_error(m.model, Xv, yv)
    manual = _rmse(m.predict(Xv), yv)
    assert internal == pytest.approx(manual, rel=1e-12, abs=1e-12)


def test_validation_pick_is_no_worse_than_mle_on_validation():
    # both fits share the identical MLE-driven Boxmin trajectory; validation only
    # changes the FINAL pick, so its held-out error can never exceed the MLE pick's.
    rng = np.random.default_rng(2)
    X = rng.random((22, 1))
    y = _fun(X) + 0.05 * rng.standard_normal(22)  # noisy -> MLE theta can overfit
    Xv = rng.random((40, 1))
    yv = _fun(Xv)  # clean held-out target

    mle = _model()
    mle.fit(X, y)
    val = _model(optimizer=Boxmin(validation=(Xv, yv)))
    val.fit(X, y)

    assert _rmse(val.predict(Xv), yv) <= _rmse(mle.predict(Xv), yv) + 1e-9


def test_validation_with_lbfgs_selects_and_stays_valid():
    # LBFGS ranks its full recorded search history by held-out error. With the same
    # seed the visited thetas are identical to the MLE run, so val selection -- which
    # also has the converged optimum in its candidate set -- is never worse on val.
    rng = np.random.default_rng(3)
    X = rng.random((25, 1))
    y = _fun(X) + 0.05 * rng.standard_normal(25)
    Xv = rng.random((40, 1))
    yv = _fun(Xv)

    mle = _model(optimizer=LBFGS(n_restarts=5))
    mle.fit(X, y)
    val = _model(optimizer=LBFGS(n_restarts=5, validation=(Xv, yv)))
    val.fit(X, y)

    theta = np.ravel(val.model["theta"])
    assert np.all((theta >= 1e-5) & (theta <= 100.0))
    assert np.all(np.isfinite(val.predict(Xv)))
    assert _rmse(val.predict(Xv), yv) <= _rmse(mle.predict(Xv), yv) + 1e-9


def test_lbfgs_validation_selects_from_history_with_no_restarts():
    # the point of recording the search history: even with n_restarts=0, a single
    # gradient descent visits many thetas, so validation selection is meaningful and
    # can land on a theta different from the descent's converged optimum.
    rng = np.random.default_rng(7)
    X = rng.random((25, 1))
    y = _fun(X) + 0.08 * rng.standard_normal(25)  # noisy -> MLE optimum overfits
    Xv = rng.random((60, 1))
    yv = _fun(Xv)

    mle = _model(optimizer=LBFGS())  # n_restarts=0, single descent
    mle.fit(X, y)
    val = _model(optimizer=LBFGS(validation=(Xv, yv)))
    val.fit(X, y)

    # the history pick is no worse on validation than the converged MLE optimum
    assert _rmse(val.predict(Xv), yv) <= _rmse(mle.predict(Xv), yv) + 1e-9
    assert np.all(np.isfinite(val.predict(Xv)))


def test_refit_carries_new_validation_via_optimizer_override():
    # refit needs no new surface: a new validation set rides the existing
    # optimizer= override, and that override is restored afterwards (per-call only).
    rng = np.random.default_rng(4)
    X0 = rng.random((15, 1))
    Xv1, Xv2 = rng.random((20, 1)), rng.random((20, 1))

    configured = Boxmin(validation=(Xv1, _fun(Xv1)))
    m = _model(optimizer=configured)
    m.fit(X0, _fun(X0))

    Xn = rng.random((8, 1))
    m.refit(Xn, _fun(Xn), optimizer=Boxmin(validation=(Xv2, _fun(Xv2))))

    assert m.optimizer is configured  # per-call override restored
    assert m.model["X"].shape[0] == 23  # data appended
    assert np.all(np.isfinite(m.predict(np.linspace(0, 1, 30)[:, None])))


def test_validation_selection_handles_matrix_Y():
    # multi-output: Y is (n, k). _val_error scores in original units across all
    # outputs, and selection still yields a valid multi-output model.
    rng = np.random.default_rng(5)
    X = rng.random((24, 2))
    Y = np.column_stack([np.sum(np.sin(X * 3.0), axis=1), np.sum(np.cos(X * 2.0), axis=1)])
    Xv = rng.random((30, 2))
    Yv = np.column_stack([np.sum(np.sin(Xv * 3.0), axis=1), np.sum(np.cos(Xv * 2.0), axis=1)])

    m = _model(optimizer=Boxmin(validation=(Xv, Yv)))
    m.fit(X, Y)

    pred = m.predict(Xv)
    assert pred.shape == (30, 2)
    assert np.all(np.isfinite(pred))
    # _val_error agrees with a manual original-space RMSE over the full (m, k) block
    assert m._val_error(m.model, Xv, Yv) == pytest.approx(_rmse(pred, Yv), rel=1e-12, abs=1e-12)
