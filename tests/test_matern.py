"""Tests for the Matérn kernel: forward properties, gradients, and end-to-end fit."""

import numpy as np
import pytest

from pydacefit.corr import Exponential, Matern
from pydacefit.dace import DACE
from pydacefit.regr import ConstantRegression


def test_invalid_nu_raises():
    # only the half-integer closed forms are supported
    with pytest.raises(ValueError, match="nu in"):
        Matern(nu=1.0)


def test_forward_is_valid_correlation():
    rng = np.random.default_rng(0)
    D = rng.uniform(-2, 2, (8, 4))
    for nu in (0.5, 1.5, 2.5):
        k = Matern(nu=nu)
        assert np.isclose(k(np.zeros((1, 4)), 0.7)[0], 1.0)  # corr is 1 at zero separation
        vals = k(D, np.array([0.7, 1.1, 0.3, 0.9]))
        assert np.all(vals > 0) and np.all(vals <= 1.0)


def test_nu_half_equals_exponential():
    # Matérn with nu=1/2 is exactly the exponential kernel exp(-theta|D|)
    rng = np.random.default_rng(1)
    D = rng.uniform(-1.5, 1.5, (6, 3))
    theta = np.array([0.7, 1.3, 0.4])
    assert np.allclose(Matern(nu=0.5)(D, theta), Exponential()(D, theta))


def test_spatial_gradient_matches_finite_differences():
    # gradient w.r.t. the design point vs central differences, away from the |D|=0 kink
    rng = np.random.default_rng(2)
    D = rng.uniform(0.3, 1.5, (6, 3)) * rng.choice([-1.0, 1.0], (6, 3))
    eps = 1e-6
    for nu in (1.5, 2.5):
        k = Matern(nu=nu)
        for theta in (0.9, np.array([0.7, 1.3, 0.4])):
            analytic = k.grad(D, theta)
            numeric = np.zeros_like(D)
            for j in range(D.shape[1]):
                dp, dm = D.copy(), D.copy()
                dp[:, j] += eps
                dm[:, j] -= eps
                numeric[:, j] = (k(dp, theta) - k(dm, theta)) / (2 * eps)
            assert np.allclose(analytic, numeric, atol=1e-6)


def test_dace_fits_and_predicts_with_matern():
    rng = np.random.default_rng(3)
    X = rng.random((22, 2))
    y = np.sum(np.sin(X * 3.0), axis=1)
    model = DACE(regr=ConstantRegression(), corr=Matern(nu=2.5), theta=0.5, thetaL=1e-3, thetaU=20.0)
    model.fit(X, y)
    pred = model.predict(rng.random((5, 2))).y
    assert pred.shape == (5, 1)
    assert np.all(np.isfinite(pred))
