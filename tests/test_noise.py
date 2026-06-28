"""Tests for the deliberate observation-noise term (DACE(noise=...))."""

import numpy as np

from pydacefit.corr import Gaussian
from pydacefit.dace import DACE
from pydacefit.regr import ConstantRegression


def _fixed(noise=0.0, theta=20.0):
    # fixed theta (no search) so the tests isolate the noise term from theta optimization;
    # max_noise=0 keeps it strict, so a clean interpolation is genuine, not auto-repaired.
    return DACE(
        regr=ConstantRegression(),
        corr=Gaussian(),
        theta=theta,
        thetaL=None,
        thetaU=None,
        noise=noise,
        max_noise=0.0,
    )


def _rmse(a, b):
    return float(np.sqrt(np.mean((np.ravel(a) - np.ravel(b)) ** 2)))


def test_noise_zero_interpolates_training_points():
    # well-separated grid + fixed theta -> noise=0 interpolates the training points exactly
    X = np.linspace(0, 1, 8)[:, None]
    y = np.sum(np.sin(X * 3.0), axis=1)

    m = _fixed(noise=0.0)
    m.fit(X, y)

    assert _rmse(m.predict(X).y, y) < 1e-6
    assert m.model["noise"] == 0.0


def test_noise_positive_smooths_instead_of_interpolating():
    # same data and theta: noise>0 is a regression GP -- it no longer passes through the
    # (noisy) training points, so its training residual is far larger than the
    # interpolating noise=0 fit. The deliberate noise is applied as-is (no climbing).
    rng = np.random.default_rng(1)
    X = np.linspace(0, 1, 20)[:, None]
    y = np.sum(np.sin(X * 3.0), axis=1) + 0.1 * rng.standard_normal(20)

    interp = _fixed(noise=0.0)
    interp.fit(X, y)
    reg = _fixed(noise=0.1)
    reg.fit(X, y)

    assert _rmse(interp.predict(X).y, y) < 1e-6  # interpolates
    assert _rmse(reg.predict(X).y, y) > 1e-2  # smooths -- visibly off the noisy points
    assert reg.model["noise"] == 0.1  # deliberate noise, applied as-is (no climbing needed)


def test_noise_present_during_the_theta_search():
    # the deliberate noise must enter the likelihood the optimizer maximizes, so a
    # noisy fit still yields a valid in-bounds theta and finite predictions.
    rng = np.random.default_rng(2)
    X = rng.random((22, 2))
    y = np.sum(np.sin(X * 3.0), axis=1) + 0.1 * rng.standard_normal(22)

    m = DACE(regr=ConstantRegression(), corr=Gaussian(), theta=1.0, thetaL=1e-3, thetaU=50.0, noise=0.05)
    m.fit(X, y)

    theta = np.ravel(m.model["theta"])
    assert np.all((theta >= 1e-3) & (theta <= 50.0))
    assert np.all(np.isfinite(m.predict(rng.random((5, 2))).y))
