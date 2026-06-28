"""Contract tests for the object-oriented kernel / trend interface."""

import numpy as np
import pytest

from pydacefit.corr import (
    Correlation,
    Cubic,
    Exponential,
    Gaussian,
    GeneralizedExponential,
    Linear,
    Matern,
    RationalQuadratic,
    Spherical,
    Spline,
)
from pydacefit.dace import DACE
from pydacefit.regr import ConstantRegression, LinearRegression, QuadraticRegression, Regression

# every correlation kernel as a ready instance (expg keeps power inside theta)
KERNELS = [
    Gaussian(),
    Cubic(),
    Exponential(),
    Linear(),
    Spherical(),
    Spline(),
    RationalQuadratic(),
    Matern(nu=0.5),
    Matern(nu=1.5),
    Matern(nu=2.5),
]
TRENDS = [ConstantRegression(), LinearRegression(), QuadraticRegression()]


def test_base_methods_raise_not_implemented():
    # the protocol's default methods are the "optional gradient" hooks
    base = Correlation()
    D, theta = np.ones((3, 2)), np.array([0.5, 0.5])
    for call in (lambda: base(D, theta), lambda: base.grad(D, theta), lambda: base.theta_grad(D, theta)):
        with pytest.raises(NotImplementedError):
            call()
    with pytest.raises(NotImplementedError):
        Regression()(D)
    with pytest.raises(NotImplementedError):
        Regression().grad(D)


def test_kernels_are_correlation_instances_and_callable():
    rng = np.random.default_rng(0)
    D = rng.uniform(-1, 1, (5, 2))
    theta = np.array([0.5, 0.5])
    for k in KERNELS:
        assert isinstance(k, Correlation)
        r = k(D, theta)
        assert r.shape == (5,)
        assert np.all(np.isfinite(r))
        # every shipped kernel also has an analytic spatial gradient
        assert k.grad(D, theta).shape == D.shape


def test_all_shipped_kernels_have_analytic_theta_grad():
    # every shipped kernel now implements an analytic theta-gradient (via the
    # _dtheta_per_dim hook, or a full theta_grad override for GeneralizedExponential).
    # A bare custom kernel that implements neither still raises, so LBFGS can detect it.
    D = np.ones((4, 2)) * 0.3
    for k in KERNELS:
        theta = np.array([0.3])  # isotropic; GeneralizedExponential needs a power too
        if isinstance(k, GeneralizedExponential):
            theta = np.array([0.3, 1.5])
        out = k.theta_grad(D, theta)
        assert out.shape[0] == D.shape[0]
        assert np.all(np.isfinite(out))

    class Bare(Correlation):
        def __call__(self, D, theta):
            return np.ones(D.shape[0])

    with pytest.raises(NotImplementedError):
        Bare().theta_grad(D, np.array([0.3]))


# theta-gradient finite-difference cases. Compact-support kernels (cubic/linear/
# spherical/spline) use small theta so |D|*theta stays well inside the support, away
# from the kinks where a finite difference would be meaningless.
_TG_CASES = [
    (Gaussian(), np.array([0.6]), np.array([0.4, 0.7, 0.5])),
    (Exponential(), np.array([0.5]), np.array([0.3, 0.6, 0.4])),
    (Cubic(), np.array([0.05]), np.array([0.04, 0.06, 0.05])),
    (Linear(), np.array([0.05]), np.array([0.04, 0.06, 0.05])),
    (Spherical(), np.array([0.05]), np.array([0.04, 0.06, 0.05])),
    (Spline(), np.array([0.05]), np.array([0.04, 0.06, 0.05])),
    (RationalQuadratic(alpha=0.7), np.array([0.6]), np.array([0.4, 0.7, 0.5])),
    (Matern(nu=1.5), np.array([0.6]), np.array([0.4, 0.7, 0.5])),
    (Matern(nu=2.5), np.array([0.6]), np.array([0.4, 0.7, 0.5])),
]


def _fd_theta(kernel, D, theta, eps=1e-6):
    theta = theta.astype(float)
    out = np.zeros((D.shape[0], len(theta)))
    for k in range(len(theta)):
        tp, tm = theta.copy(), theta.copy()
        tp[k] += eps
        tm[k] -= eps
        out[:, k] = (kernel(D, tp) - kernel(D, tm)) / (2 * eps)
    return out


@pytest.mark.parametrize("kernel,theta_iso,theta_ard", _TG_CASES, ids=[repr(c[0]) for c in _TG_CASES])
def test_theta_grad_matches_finite_difference(kernel, theta_iso, theta_ard):
    rng = np.random.default_rng(0)
    # |D| in [0.3, 1.0] with mixed signs -> nonzero (away from the abs kink at 0)
    D = rng.uniform(0.3, 1.0, (6, 3)) * rng.choice([-1.0, 1.0], (6, 3))
    for theta in (theta_iso, theta_ard):  # isotropic (1 column) and ARD (d columns)
        analytic = kernel.theta_grad(D, theta)
        numeric = _fd_theta(kernel, D, theta)
        assert analytic.shape == numeric.shape
        assert np.allclose(analytic, numeric, atol=1e-6), (repr(kernel), theta)


def test_generalized_exponential_theta_grad_matches_fd_including_power():
    # the special case: theta = (length_scale(s)..., power). The FD must match on every
    # column, INCLUDING the trailing power column that only this kernel carries.
    rng = np.random.default_rng(1)
    D = rng.uniform(0.3, 1.0, (6, 3)) * rng.choice([-1.0, 1.0], (6, 3))
    expg = GeneralizedExponential()
    for theta in (np.array([0.5, 1.6]), np.array([0.4, 0.7, 0.5, 1.6])):  # iso, ARD
        analytic = expg.theta_grad(D, theta)
        numeric = _fd_theta(expg, D, theta)
        assert analytic.shape == numeric.shape
        assert np.allclose(analytic, numeric, atol=1e-6), theta


def test_generalized_exponential_ard_uses_input_dimensionality():
    # regression: the ARD branch is keyed on the input dimensionality d (D.shape[1]),
    # not the pair count len(D). With d=3 an ARD theta is (l1, l2, l3, power) of length
    # d+1=4; the older len(D)+1 check made this raise. Compare against the equivalent
    # isotropic call (equal length-scales) which must give the same correlation.
    rng = np.random.default_rng(0)
    D = rng.uniform(-1.0, 1.0, (6, 3))
    expg = GeneralizedExponential()
    ard = expg(D, np.array([0.7, 0.7, 0.7, 1.5]))  # per-dim length-scales + power
    iso = expg(D, np.array([0.7, 1.5]))  # single length-scale + power
    assert ard.shape == (6,)
    assert np.allclose(ard, iso)
    # the analytic spatial gradient must also work (and not raise) in the ARD case
    assert expg.grad(D, np.array([0.7, 0.7, 0.7, 1.5])).shape == D.shape


def test_repr_is_readable():
    # the base repr is the class name; RationalQuadratic additionally shows its shape
    # parameter (alpha), since that is part of the object's identity.
    assert repr(Gaussian()) == "Gaussian"
    assert repr(GeneralizedExponential()) == "GeneralizedExponential"
    assert repr(ConstantRegression()) == "ConstantRegression"
    assert repr(RationalQuadratic()) == "RationalQuadratic(alpha=0.25)"
    assert repr(RationalQuadratic(alpha=1.0)) == "RationalQuadratic(alpha=1.0)"


def test_trend_grad_matches_finite_difference():
    # the analytic basis gradient (used by predict(grad=True)) vs FD.
    # grad(X) is evaluated at a single point and is constant in X for these trends.
    rng = np.random.default_rng(1)
    x = rng.random((1, 3))
    eps = 1e-6
    for trend in TRENDS:
        analytic = trend.grad(x)  # shape (d, n_basis)
        numeric = np.zeros_like(analytic)
        for k in range(x.shape[1]):
            xp, xm = x.copy(), x.copy()
            xp[0, k] += eps
            xm[0, k] -= eps
            numeric[k] = (trend(xp)[0] - trend(xm)[0]) / (2 * eps)
        assert np.allclose(analytic, numeric, atol=1e-6)


def test_dace_accepts_kernel_and_trend_instances():
    # end-to-end smoke: passing the objects directly fits and predicts finitely.
    rng = np.random.default_rng(2)
    X = rng.random((18, 2))
    y = np.sum(np.sin(X * 3.0), axis=1)
    model = DACE(regr=QuadraticRegression(), corr=RationalQuadratic(alpha=0.5), theta=0.5, thetaL=0.05, thetaU=10.0)
    model.fit(X, y)
    assert np.all(np.isfinite(model.predict(rng.random((4, 2))).y))


def test_default_kernel_and_trend_are_gaussian_constant():
    model = DACE()
    assert isinstance(model.kernel, Gaussian)
    assert isinstance(model.regr, ConstantRegression)
