"""Correctness tests: DACE predictions/MSE/gradients vs committed reference values."""

import os.path
import unittest

import numpy as np

from pydacefit.corr import (
    Cubic,
    Exponential,
    Gaussian,
    GeneralizedExponential,
    Linear,
    Spherical,
    Spline,
)
from pydacefit.dace import DACE
from pydacefit.regr import ConstantRegression, LinearRegression, QuadraticRegression


def load(name, extensions):
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "resources")

    def _load(name, suffix):
        try:
            return np.loadtxt(os.path.join(path, "%s.%s" % (name, suffix)))
        except Exception:
            return None

    return [_load(name, f) for f in extensions]


# theta bound vectors shared by the vector / ARD optimization cases
_TL = np.array([0.01, 0.01, 0.01])
_TU = np.array([20.0, 20.0, 20.0])

# (resource_name, regr, corr, theta, thetaL, thetaU). Every name has committed
# reference files under tests/resources/; thetaL/thetaU = None disables the
# hyperparameter optimization for that case.
CASES = [
    ("constant_corrgaus_1", ConstantRegression(), Gaussian(), 1.0, None, None),
    ("constant_corrgaus_opt", ConstantRegression(), Gaussian(), 0.1, 0.01, 20),
    ("linear_corrgaus_opt", LinearRegression(), Gaussian(), 0.1, 0.01, 20),
    ("quadratic_corrgauss_opt", QuadraticRegression(), Gaussian(), 0.1, 0.01, 20),
    ("constant_corrgaus_vector", LinearRegression(), Gaussian(), np.array([0.1, 0.2, 0.3]), None, None),
    (
        "constant_corrgaus_vector_opt",
        ConstantRegression(),
        Gaussian(),
        np.array([0.1, 0.2, 0.3]),
        np.array([0.01, 0.02, 0.03]),
        np.array([20, 30, 40]),
    ),
    ("constant_corrcubic_opt", ConstantRegression(), Cubic(), 0.1, 0.01, 20),
    ("constant_correxp_vector_opt", ConstantRegression(), Exponential(), np.array([1.0, 1.0, 1.0]), _TL, _TU),
    ("constant_corrlin_vector_opt", ConstantRegression(), Linear(), np.array([0.1, 0.1, 0.1]), _TL, _TU),
    (
        "constant_corrspherical_vector_opt",
        ConstantRegression(),
        Spherical(),
        np.array([0.1, 0.1, 0.1]),
        _TL,
        _TU,
    ),
    ("constant_corrspline_opt", ConstantRegression(), Spline(), 0.1, 0.01, 20),
    ("constant_correxpg_opt", ConstantRegression(), GeneralizedExponential(), np.array([0.1, 0.1]), _TL[:2], _TU[:2]),
    ("constant_corrgauss_mse", ConstantRegression(), Gaussian(), 0.1, None, None),
    ("constant_corrgauss_grad", ConstantRegression(), Gaussian(), 0.1, 0.01, 20),
    ("constant_corrlin_grad", ConstantRegression(), Linear(), 0.1, 0.01, 20),
    ("linear_corrspherical_grad", LinearRegression(), Spherical(), 0.1, 0.01, 20),
    ("quadratic_corrcubic_grad", QuadraticRegression(), Cubic(), 0.1, 0.01, 20),
    ("quadratic_correxp_grad", QuadraticRegression(), Exponential(), 0.1, 0.01, 20),
    ("quadratic_corrspline_grad", QuadraticRegression(), Spline(), 0.1, 0.01, 20),
    (
        "quadratic_correxpg_grad",
        QuadraticRegression(),
        GeneralizedExponential(),
        np.array([0.1, 0.1]),
        _TL[:2],
        _TU[:2],
    ),
    ("quadratic_corrgauss_mse_grad", QuadraticRegression(), Gaussian(), 0.1, 0.01, 20),
]


class CorrectTest(unittest.TestCase):
    def test_correctness(self):
        for name, regr, corr, theta, thetaL, thetaU in CASES:
            with self.subTest(case=name):
                X_train, F_train, X_test, correct, mse, grad = tuple(
                    load(name, ["x_train", "f_train", "x_test", "f_test", "mse", "grad"])
                )

                dacefit = DACE(regr=regr, corr=corr, theta=theta, thetaL=thetaL, thetaU=thetaU)
                dacefit.fit(X_train, F_train)

                p = dacefit.predict(X_test, mse=True, grad=True)
                pred, _mse, _grad = p.y, p.mse, p.grad

                if dacefit.tl is not None:
                    (theta_ref,) = load(name, ["theta"])
                    my_theta = np.stack([m["theta"] for m in dacefit.optimization["models"]])

                    if theta_ref.ndim == 1:
                        theta_ref = theta_ref[:, None]

                    # the full Boxmin pattern-search trajectory must match MATLAB DACE's.
                    # 1e-9 (not 1e-12) leaves headroom for cross-platform BLAS jitter along
                    # the multi-step path while still pinning every visited theta tightly.
                    self.assertEqual(len(theta_ref), len(my_theta))
                    self.assertTrue(np.all(np.abs(theta_ref - my_theta) < 1e-9))

                self.assertTrue(np.all(np.abs(correct[:, None] - pred) < 1e-6))

                if mse is not None:
                    self.assertTrue(np.all(np.abs(mse[:, None] - _mse) < 1e-6))

                if grad is not None:
                    self.assertTrue(np.all(np.abs(grad - _grad) < 1e-5))


if __name__ == "__main__":
    unittest.main()
