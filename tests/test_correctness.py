"""Correctness tests: DACE predictions/MSE/gradients vs committed reference values."""

import os.path
import unittest

import numpy as np

from pydacefit.corr import (
    corr_cubic,
    corr_exp,
    corr_expg,
    corr_gauss,
    corr_lin,
    corr_spherical,
    corr_spline,
)
from pydacefit.dace import DACE
from pydacefit.regr import regr_constant, regr_linear, regr_quadratic


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
    ("constant_corrgaus_1", regr_constant, corr_gauss, 1.0, None, None),
    ("constant_corrgaus_opt", regr_constant, corr_gauss, 0.1, 0.01, 20),
    ("linear_corrgaus_opt", regr_linear, corr_gauss, 0.1, 0.01, 20),
    ("quadratic_corrgauss_opt", regr_quadratic, corr_gauss, 0.1, 0.01, 20),
    ("constant_corrgaus_vector", regr_linear, corr_gauss, np.array([0.1, 0.2, 0.3]), None, None),
    (
        "constant_corrgaus_vector_opt",
        regr_constant,
        corr_gauss,
        np.array([0.1, 0.2, 0.3]),
        np.array([0.01, 0.02, 0.03]),
        np.array([20, 30, 40]),
    ),
    ("constant_corrcubic_opt", regr_constant, corr_cubic, 0.1, 0.01, 20),
    ("constant_correxp_vector_opt", regr_constant, corr_exp, np.array([1.0, 1.0, 1.0]), _TL, _TU),
    ("constant_corrlin_vector_opt", regr_constant, corr_lin, np.array([0.1, 0.1, 0.1]), _TL, _TU),
    (
        "constant_corrspherical_vector_opt",
        regr_constant,
        corr_spherical,
        np.array([0.1, 0.1, 0.1]),
        _TL,
        _TU,
    ),
    ("constant_corrspline_opt", regr_constant, corr_spline, 0.1, 0.01, 20),
    ("constant_correxpg_opt", regr_constant, corr_expg, np.array([0.1, 0.1]), _TL[:2], _TU[:2]),
    ("constant_corrgauss_mse", regr_constant, corr_gauss, 0.1, None, None),
    ("constant_corrgauss_grad", regr_constant, corr_gauss, 0.1, 0.01, 20),
    ("constant_corrlin_grad", regr_constant, corr_lin, 0.1, 0.01, 20),
    ("linear_corrspherical_grad", regr_linear, corr_spherical, 0.1, 0.01, 20),
    ("quadratic_corrcubic_grad", regr_quadratic, corr_cubic, 0.1, 0.01, 20),
    ("quadratic_correxp_grad", regr_quadratic, corr_exp, 0.1, 0.01, 20),
    ("quadratic_corrspline_grad", regr_quadratic, corr_spline, 0.1, 0.01, 20),
    ("quadratic_correxpg_grad", regr_quadratic, corr_expg, np.array([0.1, 0.1]), _TL[:2], _TU[:2]),
    ("quadratic_corrgauss_mse_grad", regr_quadratic, corr_gauss, 0.1, 0.01, 20),
]


class CorrectTest(unittest.TestCase):
    def test_correctness(self):
        for name, regr, corr, theta, thetaL, thetaU in CASES:
            with self.subTest(case=name):
                X_train, F_train, X_test, correct, mse, grad, mse_grad = tuple(
                    load(name, ["x_train", "f_train", "x_test", "f_test", "mse", "grad", "grad_mse"])
                )

                dacefit = DACE(regr=regr, corr=corr, theta=theta, thetaL=thetaL, thetaU=thetaU)
                dacefit.fit(X_train, F_train)

                pred, _mse, _grad, _mse_grad = dacefit.predict(
                    X_test, return_mse=True, return_gradient=True, return_mse_gradient=True
                )

                if dacefit.tl is not None:
                    (theta_ref,) = load(name, ["theta"])
                    my_theta = np.stack([m["theta"] for m in dacefit.itpar["models"]])

                    if theta_ref.ndim == 1:
                        theta_ref = theta_ref[:, None]

                    self.assertEqual(len(theta_ref), len(my_theta))
                    self.assertTrue(np.all(np.abs(theta_ref - my_theta) < 1e-12))

                self.assertTrue(np.all(np.abs(correct[:, None] - pred) < 1e-6))

                if mse is not None:
                    self.assertTrue(np.all(np.abs(mse[:, None] - _mse) < 1e-6))

                if grad is not None:
                    self.assertTrue(np.all(np.abs(grad - _grad) < 1e-5))


if __name__ == "__main__":
    unittest.main()
