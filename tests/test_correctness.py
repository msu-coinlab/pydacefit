import unittest

import os.path

import numpy as np

from pydacefit.corr import corr_gauss
from pydacefit.dace import DACE
from pydacefit.regr import regr_constant, regr_linear, regr_quadratic


def load(name, extensions):
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "resources")

    def _load(name, suffix):
        return np.loadtxt(os.path.join(path, "%s.%s" % (name, suffix)))

    return [_load(name, f) for f in extensions]


class CorrectTest(unittest.TestCase):

    def test_correctness(self):
        tests = [
            ("constant_corrgaus_1", DACE(regr=regr_constant, corr=corr_gauss, theta=1.0)),
            ("constant_corrgaus_opt", DACE(regr=regr_constant, corr=corr_gauss, theta=0.1, tl=0.01, tu=20)),
            ("linear_corrgaus_opt", DACE(regr=regr_linear, corr=corr_gauss, theta=0.1, tl=0.01, tu=20)),
            ("quadratic_corrgaus_opt", DACE(regr=regr_quadratic, corr=corr_gauss, theta=0.1, tl=0.01, tu=20)),
        ]

        for (name, dacefit) in tests:
            X_train, F_train, X_test, correct = tuple(load(name, ["x_train", "f_train", "x_test", "f_test"]))
            dacefit.fit(X_train, F_train)
            pred = dacefit.predict(X_test)

            if dacefit.tl is not None:
                theta, = load(name, ["theta"])
                my_theta = np.stack([m["theta"] for m in dacefit.itpar["models"]])

                is_equal = len(theta) == len(my_theta) and np.all(np.abs(theta[:, None] - my_theta) < 1e-12)

                if not is_equal:
                    dacefit.fit(X_train, F_train)

                self.assertTrue(is_equal)

            self.assertTrue(np.all(np.abs(correct[:, None] - pred) < 0.00001))


if __name__ == '__main__':
    unittest.main()
