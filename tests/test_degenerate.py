"""Contract tests for degenerate / ill-conditioned inputs.

These pin behavior *contracts* — does a case raise, return finite, or return the
documented NaN placeholder — rather than tight numeric snapshots. Ill-conditioned
outputs are numerically unstable (the same sensitivity that makes different linear
solvers diverge on them), so they are unfit for a tight cross-platform golden
baseline; a contract is the right, portable thing to assert instead.
"""

import numpy as np
import pytest

from pydacefit.corr import corr_gauss
from pydacefit.dace import DACE
from pydacefit.regr import regr_constant, regr_quadratic


def _model(regr=regr_constant, theta=0.5, thetaL=None, thetaU=None):
    return DACE(regr=regr, corr=corr_gauss, theta=theta, thetaL=thetaL, thetaU=thetaU)


def _line(n=10):
    x = np.linspace(0, 1, n)[:, None]
    return x, np.sin(x[:, 0] * 6.0)


# --- hard contracts: must fail loudly via the code-level guards ---


def test_mismatched_rows_raises():
    x, y = _line()
    with pytest.raises(Exception, match="same number of rows"):
        _model().fit(x, y[:5])


def test_mse_gradient_without_prerequisites_raises():
    x, y = _line()
    model = _model()
    model.fit(x, y)
    with pytest.raises(Exception, match="gradient and MSE"):
        model.predict(np.array([[0.5]]), return_mse_gradient=True)


def test_single_sample_is_not_silently_fit():
    # one point cannot support a regression + GP fit; the exact failure mode is
    # BLAS-dependent, so only assert it fails loudly instead of returning a number.
    with pytest.raises(Exception):
        model = _model()
        model.fit(np.zeros((1, 1)), np.zeros(1))
        model.predict(np.array([[0.5]]))


# --- documented "not implemented": the MSE-gradient is a NaN placeholder ---


def test_mse_gradient_is_nan_placeholder():
    x, y = _line()
    model = _model()
    model.fit(x, y)
    x_test = np.array([[0.3], [0.6]])
    _, _, _, mse_grad = model.predict(x_test, return_mse=True, return_gradient=True, return_mse_gradient=True)
    assert mse_grad.shape == (2, 1)
    assert np.all(np.isnan(mse_grad))


# --- robustness: near-singular but jitter-stabilized -> finite, correct shape ---


def _case_duplicate_points():
    x, y = _line()
    X = np.vstack([x, x[:1]])
    Y = np.concatenate([y, y[:1]])
    model = _model()
    model.fit(X, Y)
    return model, np.linspace(0, 1, 4)[:, None]


def _case_collinear_quadratic():
    X = np.linspace(0, 1, 12)[:, None] * np.ones((1, 2))
    y = np.sin(X[:, 0] * 6.0)
    model = _model(regr=regr_quadratic)
    model.fit(X, y)
    return model, np.linspace(0, 1, 4)[:, None] * np.ones((1, 2))


def _case_tiny_theta():
    x, y = _line()
    model = _model(theta=1e-6)
    model.fit(x, y)
    return model, np.linspace(0, 1, 4)[:, None]


@pytest.mark.parametrize(
    "build",
    [_case_duplicate_points, _case_collinear_quadratic, _case_tiny_theta],
    ids=["duplicate_points", "collinear_quadratic", "tiny_theta"],
)
def test_ill_conditioned_returns_finite(build):
    model, x_test = build()
    pred = model.predict(x_test)
    assert pred.shape == (x_test.shape[0], 1)
    assert np.all(np.isfinite(pred))
