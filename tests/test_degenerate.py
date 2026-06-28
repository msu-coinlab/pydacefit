"""Contract tests for degenerate / ill-conditioned inputs.

These pin behavior *contracts* — does a case raise, return finite, or return the
documented NaN placeholder — rather than tight numeric snapshots. Ill-conditioned
outputs are numerically unstable (the same sensitivity that makes different linear
solvers diverge on them), so they are unfit for a tight cross-platform golden
baseline; a contract is the right, portable thing to assert instead.
"""

import numpy as np
import pytest

from pydacefit.corr import Gaussian
from pydacefit.dace import DACE
from pydacefit.fit import DaceFitError
from pydacefit.regr import ConstantRegression, QuadraticRegression


def _model(regr=ConstantRegression(), theta=0.5, thetaL=None, thetaU=None):
    return DACE(regr=regr, corr=Gaussian(), theta=theta, thetaL=thetaL, thetaU=thetaU)


def _line(n=10):
    x = np.linspace(0, 1, n)[:, None]
    return x, np.sin(x[:, 0] * 6.0)


# --- hard contracts: must fail loudly via the code-level guards ---


def test_mismatched_rows_raises():
    x, y = _line()
    with pytest.raises(Exception, match="same number of rows"):
        _model().fit(x, y[:5])


def test_single_sample_is_not_silently_fit():
    # one point cannot support a regression + GP fit; the exact failure mode is
    # BLAS-dependent, so only assert it fails loudly instead of returning a number.
    with pytest.raises(Exception):
        model = _model()
        model.fit(np.zeros((1, 1)), np.zeros(1))
        model.predict(np.array([[0.5]])).y


# --- zero-variance data: degrade gracefully to a constant, never silent NaN ---


def test_constant_target_returns_the_constant():
    # a constant Y has zero std -> normalization used to divide by zero and silently
    # return NaN. It must now degrade to a constant predictor: the constant everywhere.
    x, _ = _line()
    y = np.full(x.shape[0], 3.5)
    model = _model()
    model.fit(x, y)
    pred = model.predict(np.linspace(0, 1, 5)[:, None]).y
    assert np.all(np.isfinite(pred))
    assert np.allclose(pred, 3.5)


def test_constant_input_column_does_not_poison_the_fit():
    # a constant X column has zero variance too; it must not produce NaN
    x = np.linspace(0, 1, 10)[:, None]
    X = np.hstack([x, np.full((10, 1), 0.5)])  # second column constant
    y = np.sin(x[:, 0] * 6.0)
    model = _model()
    model.fit(X, y)
    Xt = np.hstack([np.linspace(0, 1, 4)[:, None], np.full((4, 1), 0.5)])
    assert np.all(np.isfinite(model.predict(Xt).y))


# --- robustness: near-singular but jitter-stabilized -> finite, correct shape ---


def _case_duplicate_points():
    x, y = _line()
    X = np.vstack([x, x[:1]])
    Y = np.concatenate([y, y[:1]])
    model = _model()
    model.fit(X, Y)
    return model, np.linspace(0, 1, 4)[:, None]


def test_collinear_quadratic_design_raises():
    # both input columns are identical, so the quadratic basis has duplicate columns
    # (rank 3 of 6) -- a genuinely singular regression design. There is no meaningful
    # GLS fit, so fit() must raise (matching MATLAB DACE's rcond(G) guard) rather than
    # silently return finite-but-garbage predictions.
    X = np.linspace(0, 1, 12)[:, None] * np.ones((1, 2))
    y = np.sin(X[:, 0] * 6.0)
    with pytest.raises(DaceFitError, match="too ill conditioned"):
        _model(regr=QuadraticRegression()).fit(X, y)


def _case_tiny_theta():
    x, y = _line()
    model = _model(theta=1e-6)
    model.fit(x, y)
    return model, np.linspace(0, 1, 4)[:, None]


@pytest.mark.parametrize(
    "build",
    [_case_duplicate_points, _case_tiny_theta],
    ids=["duplicate_points", "tiny_theta"],
)
def test_ill_conditioned_returns_finite(build):
    model, x_test = build()
    pred = model.predict(x_test).y
    assert pred.shape == (x_test.shape[0], 1)
    assert np.all(np.isfinite(pred))
