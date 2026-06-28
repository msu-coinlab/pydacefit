"""ProductKernel base: a custom subclass needs only _factor to get correct gradients."""

import numpy as np

from pydacefit.corr import ProductKernel, calc_kernel_matrix


class _Tent(ProductKernel):
    # toy compact-support factor M(t) = max(1 - t, 0), t = theta * |D| (a Linear clone),
    # defined purely through _factor -- grad / theta_grad come from the base class.
    def _factor(self, D, theta):
        ss = np.maximum(1 - np.abs(D) * theta, 0)
        return ss, np.where(ss > 0, -1.0, 0.0)


def test_custom_product_kernel_gradients_match_finite_difference():
    ker = _Tent()
    rng = np.random.default_rng(0)
    A, B = rng.random((1, 3)), rng.random((5, 3))
    theta = np.array([0.7, 1.4, 0.5])  # ARD
    D = np.repeat(A, B.shape[0], 0) - np.tile(B, (A.shape[0], 1))

    # point gradient vs central FD
    ana = ker.grad(D, theta)
    fd = np.zeros_like(ana)
    eps = 1e-7
    for k in range(3):
        Ap, Am = A.copy(), A.copy()
        Ap[0, k] += eps
        Am[0, k] -= eps
        fd[:, k] = (calc_kernel_matrix(Ap, B, ker, theta).ravel() - calc_kernel_matrix(Am, B, ker, theta).ravel()) / (
            2 * eps
        )
    assert np.allclose(ana, fd, atol=1e-5)

    # theta gradient (per-dimension, ARD) vs central FD
    tg = ker.theta_grad(D, theta)
    fd_t = np.zeros_like(tg)
    for k in range(3):
        tp, tm = theta.copy(), theta.copy()
        tp[k] += eps
        tm[k] -= eps
        fd_t[:, k] = (ker(D, tp) - ker(D, tm)) / (2 * eps)
    assert np.allclose(tg, fd_t, atol=1e-5)
    assert ker.has_theta_grad is True


def test_product_kernel_isotropic_theta_grad_collapses_to_one_column():
    # a scalar (isotropic) theta yields a single summed theta-gradient column, like the
    # built-in product kernels (the base Correlation.theta_grad handles the collapse).
    ker = _Tent()
    rng = np.random.default_rng(1)
    A, B = rng.random((1, 3)), rng.random((4, 3))
    D = np.repeat(A, B.shape[0], 0) - np.tile(B, (A.shape[0], 1))
    assert ker.theta_grad(D, 0.8).shape == (4, 1)
