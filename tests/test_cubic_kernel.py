"""Regression test for the cubic-kernel non-positive-definite Cholesky crash.

On standardized data the cubic correlation matrix is genuinely non-positive-definite at
the boxmin *start* theta (min eigenvalue ~ -0.012, far below what the tiny fit() jitter
can repair). The start fit must relocate to a feasible theta rather than crashing in
Cholesky, since larger thetas (5, 10, 50, 100) give a perfectly positive-definite matrix
that the search could have used.
"""

import numpy as np
import pytest

from pydacefit.corr import Cubic
from pydacefit.dace import DACE
from pydacefit.regr import LinearRegression


def _cubic_dataset():
    # fixed 40-point, 2-D sample from a smooth function (matches the repro)
    rng = np.random.default_rng(42)
    X = rng.random((40, 2))
    y = (np.sin(3 * X[:, 0]) + np.cos(2 * X[:, 1])).reshape(-1, 1)
    return X, y


def test_cubic_kernel_fits_despite_non_pd_start_theta():
    # the cubic correlation matrix is non-PD at the default start theta (=1.0).
    # boxmin must relocate to a feasible (positive-definite) theta and fit EXACTLY
    # (no added noise), rather than crashing in Cholesky.
    from pydacefit.corr import calc_kernel_matrix

    X, y = _cubic_dataset()
    model = DACE(regr=LinearRegression(), corr=Cubic(), theta=1.0, thetaL=1e-5, thetaU=100.0)
    model.fit(X, y)

    assert np.all(np.isfinite(model.predict(X).y))
    assert model.model["noise"] == 0.0  # relocation found a feasible theta -> exact fit

    # the theta it settled on must give a genuinely positive-definite matrix
    nX = (X - X.mean(0)) / X.std(0, ddof=1)
    R = calc_kernel_matrix(nX, nX, Cubic(), theta=model.model["theta"])
    assert np.linalg.eigvalsh(R).min() > -1e-8


def test_no_feasible_theta_raises_by_default():
    # [0.5, 1.0] is an entirely non-PD bracket for cubic on this data, needing ~1.2%
    # noise -- far above the default max_noise=1e-4 numerical-repair ceiling -- so the
    # default must still surface the infeasibility loudly.
    X, y = _cubic_dataset()
    model = DACE(regr=LinearRegression(), corr=Cubic(), theta=0.7, thetaL=0.5, thetaU=1.0)
    with pytest.raises(Exception, match="positive-definite"):
        model.fit(X, y)


def test_no_feasible_theta_climbs_within_max_noise_ceiling():
    # with a generous max_noise the same case must not crash: it climbs the noise to the
    # smallest amount that makes R positive-definite (with a warning), records it, predicts.
    X, y = _cubic_dataset()
    model = DACE(regr=LinearRegression(), corr=Cubic(), theta=0.7, thetaL=0.5, thetaU=1.0, max_noise=1.0)
    with pytest.warns(UserWarning, match="noise"):
        model.fit(X, y)
    assert np.all(np.isfinite(model.predict(X).y))
    noise = model.model["noise"]
    assert 0.0 < noise <= 1.0  # regularized, within ceiling


def test_validation_split_keeps_both_train_and_full_spd():
    # the SPD requirement applies to TWO matrices: R over the train rows (built during
    # the search) and R over all rows (the append=True final fit). On this all-non-PD
    # cubic bracket, with a validation mask and a max_noise ceiling, BOTH are
    # regularized to positive-definiteness independently -- the search anchors a
    # feasible train model, and the final all-rows fit climbs its own noise.
    from pydacefit.corr import calc_kernel_matrix

    X, y = _cubic_dataset()
    mask = np.zeros(X.shape[0], dtype=bool)
    mask[::4] = True  # ~1/4 held out for theta selection

    model = DACE(regr=LinearRegression(), corr=Cubic(), theta=0.7, thetaL=0.5, thetaU=1.0, max_noise=1.0)
    with pytest.warns(UserWarning, match="noise"):
        model.fit(X, y, validation=mask)

    # the committed all-rows model is SPD (its recorded noise made it so)
    assert model.model["noise"] > 0.0
    np.linalg.cholesky(model.model["R"])  # raises if not SPD

    # and the noise was genuinely necessary: the raw full correlation matrix at the
    # chosen theta is non-SPD without it
    nX = model.model["nX"]
    R_raw = calc_kernel_matrix(nX, nX, Cubic(), theta=model.model["theta"])
    assert np.linalg.eigvalsh(R_raw).min() < 0.0
    assert np.all(np.isfinite(model.predict(X).y))


def test_repair_climbs_independent_of_deliberate_noise():
    # the repair budget (max_noise) is ADDED ON TOP of the deliberate noise, not an
    # absolute ceiling: even when max_noise < noise, a non-PD R is still repaired by
    # climbing above noise. (An absolute-ceiling design raised immediately here, because
    # the first climb step already exceeds a max_noise that sits below noise.)
    from pydacefit.corr import calc_kernel_matrix
    from pydacefit.fit import fit

    X, y = _cubic_dataset()
    nX = (X - X.mean(0)) / X.std(0, ddof=1)
    nY = (y - y.mean(0)) / y.std(0, ddof=1)
    theta = 0.7
    deficit = -np.linalg.eigvalsh(calc_kernel_matrix(nX, nX, Cubic(), theta)).min()
    assert deficit > 0  # genuinely non-PD at this theta

    noise = deficit * 0.8  # deliberate noise just under the deficit -> still non-PD
    max_noise = deficit * 0.5  # BELOW noise (the regime that broke an absolute ceiling)
    with pytest.warns(UserWarning, match="repair"):
        model = fit(nX, nY, LinearRegression(), Cubic(), theta, noise=noise, max_noise=max_noise)

    assert max_noise < noise  # confirm we are in the previously-broken regime
    assert model["noise"] > noise  # repair was climbed on top of the deliberate noise
    np.linalg.cholesky(model["R"])  # and the result is positive-definite


def test_max_noise_too_small_still_raises():
    # a ceiling below what the bracket needs (min eigenvalue ~ -0.012) cannot restore
    # positive-definiteness, so it stops and raises -- the ceiling is a hard limit.
    X, y = _cubic_dataset()
    model = DACE(regr=LinearRegression(), corr=Cubic(), theta=0.7, thetaL=0.5, thetaU=1.0, max_noise=1e-6)
    with pytest.raises(Exception, match="positive-definite"):
        model.fit(X, y)


def test_cubic_correlation_matrix_is_not_pd_at_start_theta():
    # pin the root cause itself: the standardized cubic R at theta=1.0 has a clearly
    # negative eigenvalue, so this is a real non-PD matrix, not float jitter.
    from pydacefit.corr import calc_kernel_matrix

    X, _ = _cubic_dataset()
    nX = (X - X.mean(0)) / X.std(0, ddof=1)
    R = calc_kernel_matrix(nX, nX, Cubic(), theta=1.0)
    assert np.linalg.eigvalsh(R).min() < -1e-3
