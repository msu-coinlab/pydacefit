"""Regression test for the cubic-kernel non-positive-definite Cholesky crash.

Found via repro_cubic_cholesky.py. On standardized data the cubic correlation
matrix is genuinely non-positive-definite at the boxmin *start* theta (min
eigenvalue ~ -0.012, far below what the tiny fit() nugget can repair). Because
``boxmin.start`` -- unlike ``evaluate_and_set_best`` -- does not catch the Cholesky
failure, the whole fit crashes, even though larger thetas (5, 10, 50, 100) give a
perfectly positive-definite matrix that the search could have used.
"""

import numpy as np
import pytest

from pydacefit.corr import corr_cubic
from pydacefit.dace import DACE
from pydacefit.regr import regr_linear


def _cubic_dataset():
    # fixed 40-point, 2-D sample from a smooth function (matches the repro)
    rng = np.random.default_rng(42)
    X = rng.random((40, 2))
    y = (np.sin(3 * X[:, 0]) + np.cos(2 * X[:, 1])).reshape(-1, 1)
    return X, y


def test_cubic_kernel_fits_despite_non_pd_start_theta():
    # the cubic correlation matrix is non-PD at the default start theta (=1.0).
    # boxmin must relocate to a feasible (positive-definite) theta and fit, rather
    # than crashing in Cholesky. Regression for repro_cubic_cholesky.py.
    from pydacefit.corr import calc_kernel_matrix

    X, y = _cubic_dataset()
    model = DACE(regr=regr_linear, corr=corr_cubic, theta=1.0, thetaL=1e-5, thetaU=100.0)
    model.fit(X, y)

    assert np.all(np.isfinite(model.predict(X)))

    # the theta it settled on must give a genuinely positive-definite matrix
    nX = (X - X.mean(0)) / X.std(0, ddof=1)
    R = calc_kernel_matrix(nX, nX, corr_cubic, theta=model.model["theta"])
    assert np.linalg.eigvalsh(R).min() > -1e-8


def test_no_feasible_theta_raises_by_default():
    # [0.5, 1.0] is an entirely non-PD bracket for cubic on this data, so the search
    # finds no feasible theta. Default raise_error=True must surface that loudly.
    X, y = _cubic_dataset()
    model = DACE(regr=regr_linear, corr=corr_cubic, theta=0.7, thetaL=0.5, thetaU=1.0)
    with pytest.raises(Exception, match="positive-definite"):
        model.fit(X, y)


def test_no_feasible_theta_falls_back_when_raise_error_false():
    # with raise_error=False the same case must not crash: it falls back to a
    # regularized midpoint model (with a warning) so downstream still gets a fit.
    X, y = _cubic_dataset()
    model = DACE(regr=regr_linear, corr=corr_cubic, theta=0.7, thetaL=0.5, thetaU=1.0, raise_error=False)
    with pytest.warns(UserWarning, match="nugget"):
        model.fit(X, y)
    assert np.all(np.isfinite(model.predict(X)))


def test_cubic_correlation_matrix_is_not_pd_at_start_theta():
    # pin the root cause itself: the standardized cubic R at theta=1.0 has a clearly
    # negative eigenvalue, so this is a real non-PD matrix, not float jitter.
    from pydacefit.corr import calc_kernel_matrix

    X, _ = _cubic_dataset()
    nX = (X - X.mean(0)) / X.std(0, ddof=1)
    R = calc_kernel_matrix(nX, nX, corr_cubic, theta=1.0)
    assert np.linalg.eigvalsh(R).min() < -1e-3
