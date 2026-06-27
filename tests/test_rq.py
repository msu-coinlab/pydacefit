"""Tests for the Rational Quadratic kernel: forward properties + analytic gradient."""

import numpy as np

from pydacefit.corr import RationalQuadratic, corr_gauss, corr_rq


def test_forward_is_valid_correlation():
    rng = np.random.RandomState(0)
    D = rng.uniform(-2, 2, (8, 4))
    # correlation is 1 at zero separation and stays within (0, 1]
    assert np.isclose(corr_rq(np.zeros((1, 4)), 0.7)[0], 1.0)
    vals = corr_rq(D, np.array([0.7, 1.1, 0.3, 0.9]))
    assert np.all(vals > 0)
    assert np.all(vals <= 1.0)


def test_large_alpha_approaches_gauss():
    # alpha -> inf recovers the Gaussian kernel
    rng = np.random.RandomState(2)
    D = rng.uniform(-1, 1, (5, 3))
    theta = np.array([0.6, 1.2, 0.4])
    assert np.allclose(RationalQuadratic(alpha=1e6)(D, theta), corr_gauss(D, theta), atol=1e-4)


def test_gradient_matches_finite_differences():
    rng = np.random.RandomState(1)
    D = rng.uniform(-1.5, 1.5, (6, 3))
    eps = 1e-6
    for alpha in (0.25, 1.0, 3.0):
        kernel = RationalQuadratic(alpha=alpha)
        for theta in (0.9, np.array([0.7, 1.3, 0.4])):  # isotropic scalar and ARD vector
            analytic = kernel.grad(D, theta)
            numeric = np.zeros_like(D)
            for k in range(D.shape[1]):
                d_plus = D.copy()
                d_plus[:, k] += eps
                d_minus = D.copy()
                d_minus[:, k] -= eps
                numeric[:, k] = (kernel(d_plus, theta) - kernel(d_minus, theta)) / (2 * eps)
            assert np.allclose(analytic, numeric, atol=1e-6)
