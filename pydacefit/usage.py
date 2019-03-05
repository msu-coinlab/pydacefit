import numpy as np

from pydacefit.corr import corr_gauss
from pydacefit.dace import DACE
from pydacefit.regr import regr_constant

import matplotlib.pyplot as plt


def fun(X):
    return np.sum(np.sin(X * 2 * np.pi), axis=1)


X = np.random.random((20, 1))
F = fun(X)

# create
dacefit = DACE(regr=regr_constant, corr=corr_gauss, theta=1.0, tl=0.00001, tu=100)
dacefit.fit(X, F)

_X = np.linspace(0, 1, 100)[:, None]
_F = dacefit.predict(_X)

plt.scatter(X, F, label="prediction")
plt.plot(_X, _F, label="data")
plt.legend()
plt.show()

print("MSE: ", np.mean(np.abs(fun(_X)[:,None] - _F)))
