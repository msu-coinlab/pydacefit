import unittest

import os.path

import numpy as np

from pydacefit.corr import corr_gauss, corr_cubic, corr_exp, corr_lin, corr_spherical
from pydacefit.dace import DACE
from pydacefit.regr import regr_constant, regr_linear, regr_quadratic


def load(name, extensions):
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "resources")

    def _load(name, suffix):
        try:
            return np.loadtxt(os.path.join(path, "%s.%s" % (name, suffix)))
        except:
            return None

    return [_load(name, f) for f in extensions]


class CorrectTest(unittest.TestCase):

    def test_correctness(self):

        tl = np.array([0.01, 0.01, 0.01])
        tu = np.array([20, 20, 20])

        tests = [

           # ("constant_corrgaus_1", DACE(regr=regr_constant, corr=corr_gauss, theta=1.0, tl=None, tu=None)),
           # ("constant_corrgaus_opt", DACE(regr=regr_constant, corr=corr_gauss, theta=0.1, tl=0.01, tu=20)),
           # ("linear_corrgaus_opt", DACE(regr=regr_linear, corr=corr_gauss, theta=0.1, tl=0.01, tu=20)),
           ("quadratic_corrgauss_opt", DACE(regr=regr_quadratic, corr=corr_gauss, theta=0.1, tl=0.01, tu=20)),
           # ("constant_corrgaus_vector", DACE(regr=regr_linear, corr=corr_gauss, theta=np.array([0.1, 0.2, 0.3]), tl=None, tu=None)),
           # ("constant_corrgaus_vector_opt", DACE(regr=regr_constant, corr=corr_gauss, theta=np.array([0.1, 0.2, 0.3]), tl=np.array([0.01, 0.02, 0.03]), tu=np.array([20, 30, 40]))),
           # ("constant_corrcubic_opt", DACE(regr=regr_constant, corr=corr_cubic, theta=0.1, tl=0.01, tu=20)),
           # ("constant_correxp_vector_opt", DACE(regr=regr_constant, corr=corr_exp, theta=np.array([1.0, 1.0, 1.0]), tl=tl, tu=tu)),
           # ("constant_corrlin_vector_opt", DACE(regr=regr_constant, corr=corr_lin, theta=np.array([0.1, 0.1, 0.1]), tl=tl, tu=tu)),
           # ("constant_corrspherical_vector_opt", DACE(regr=regr_constant, corr=corr_spherical, theta=np.array([0.1, 0.1, 0.1]), tl=tl, tu=tu))

           #("constant_corrgauss_mse", DACE(regr=regr_constant, corr=corr_gauss, theta=0.1, tl=None, tu=None)),

        ]

        for (name, dacefit) in tests:

            print(name)

            X_train, F_train, X_test, correct, mse = tuple(load(name, ["x_train", "f_train", "x_test", "f_test", "mse"]))
            dacefit.fit(X_train, F_train)
            pred, _mse = dacefit.predict(X_test, return_mse=True)

            if dacefit.tl is not None:
                theta, = load(name, ["theta"])
                my_theta = np.stack([m["theta"] for m in dacefit.itpar["models"]])

                if len(theta.shape) == 1:
                    theta = theta[:, None]

                is_equal = len(theta) == len(my_theta) and np.all(np.abs(theta - my_theta) < 1e-12)

                self.assertTrue(is_equal)

            self.assertTrue(np.all(np.abs(correct[:, None] - pred) < 0.00001))

            if mse is not None:
                self.assertTrue(np.all(np.abs(mse[:, None] - _mse) < 0.00001))


if __name__ == '__main__':
    unittest.main()
