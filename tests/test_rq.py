"""Tests for the Rational Quadratic kernel: forward properties + analytic gradients."""

import numpy as np

from pydacefit.corr import Gaussian, RationalQuadratic

GAUSS = Gaussian()
RQ = RationalQuadratic()  # alpha=0.25 default


def test_forward_is_valid_correlation():
    rng = np.random.RandomState(0)
    D = rng.uniform(-2, 2, (8, 4))
    # correlation is 1 at zero separation and stays within (0, 1]
    assert np.isclose(RQ(np.zeros((1, 4)), 0.7)[0], 1.0)
    vals = RQ(D, np.array([0.7, 1.1, 0.3, 0.9]))
    assert np.all(vals > 0)
    assert np.all(vals <= 1.0)


def test_large_alpha_approaches_gauss():
    # alpha -> inf recovers the Gaussian kernel
    rng = np.random.RandomState(2)
    D = rng.uniform(-1, 1, (5, 3))
    theta = np.array([0.6, 1.2, 0.4])
    assert np.allclose(RationalQuadratic(alpha=1e6)(D, theta), GAUSS(D, theta), atol=1e-4)


def test_gradient_matches_finite_differences():
    # spatial gradient (w.r.t. the design point) vs central differences in D
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


def test_theta_gradient_matches_finite_differences():
    # theta gradient (used by LBFGS) vs central differences in theta. Covers both the
    # isotropic (length-1 theta, summed over dims) and ARD (per-dim) branches.
    rng = np.random.RandomState(3)
    D = rng.uniform(-1.5, 1.5, (7, 3))
    eps = 1e-6
    for alpha in (0.25, 1.0, 3.0):
        kernel = RationalQuadratic(alpha=alpha)
        for theta in (np.array([0.8]), np.array([0.7, 1.3, 0.4])):
            analytic = kernel.theta_grad(D, theta)
            numeric = np.zeros((D.shape[0], len(theta)))
            for k in range(len(theta)):
                tp, tm = theta.astype(float).copy(), theta.astype(float).copy()
                tp[k] += eps
                tm[k] -= eps
                numeric[:, k] = (kernel(D, tp) - kernel(D, tm)) / (2 * eps)
            assert analytic.shape == numeric.shape
            assert np.allclose(analytic, numeric, atol=1e-6)
