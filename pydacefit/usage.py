import numpy as np

from pydacefit.corr import corr_gauss
from pydacefit.dace import DACE
from pydacefit.regr import regr_constant

X = np.random.random(100, 1)
F = np.sum(np.sin(X), axis=1)

# create
dacefit = DACE(regr=regr_constant, corr=corr_gauss, theta=1.0, tl=0.00001, tu=100)
dacefit.fit(X, F)

_X = np.random.random(100, 2)
_F = dacefit.predict(_X)


