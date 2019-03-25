import numpy as np

from pydacefit.boxmin import start, explore, move
from pydacefit.corr import corr_gauss, calc_kernel_matrix
from pydacefit.fit import fit
from pydacefit.regr import regr_constant


class DACE:

    def __init__(self, regr=regr_constant, corr=corr_gauss, theta=1.0, tl=0.0, tu=100.0):
        """

        Parameters
        ----------
        regr : callable
            Type of regression that should be used: regr_constant, regr_linear or regr_quadratic

        corr : callable
            Type of correlation (kernel) that should be used. default: corr_gauss

        theta : float
            Initial value of theta. Can be a vector or a float

        tl : float
            The lower bound if theta should be optimized.

        tu : float
            The upper bound if theta should be optimized.

        """

        super().__init__()
        self.regr = regr
        self.kernel = corr

        # most of the model will be stored here
        self.model = None

        # the hyperparameter can be defined
        self.theta = theta

        # lower and upper bound if it should be optimized
        self.tl = tl
        self.tu = tu

        # intermediate steps saved during hyperparameter optimization
        self.itpar = None

    def fit(self, X, Y):

        # the targets should be a 2d array
        if len(Y.shape) == 1:
            Y = Y[:, None]

        # check if for each observation a target values exist
        if X.shape[0] != Y.shape[0]:
            raise Exception("X and Y must have the same number of rows.")

        # save the mean and standard deviation of the input
        mX, sX = np.mean(X, axis=0), np.std(X, axis=0, ddof=1)
        mY, sY = np.mean(Y, axis=0), np.std(Y, axis=0, ddof=1)

        # standardize the input
        nX = (X - mX) / sX
        nY = (Y - mY) / sY

        # check the hyperparamters
        if self.tl is not None and self.tu is not None:
            self.model = {'nX': nX, 'nY': nY}
            self.boxmin()
            self.model = self.itpar["best"]
        else:
            self.model = fit(nX, nY, self.regr, self.kernel, self.theta)

        self.model = {**self.model, 'mX': mX, 'sX': sX, 'mY': mY, 'sY': sY, 'nX': nX, 'nY': nY}
        self.model['sigma2'] = np.square(sY) @ self.model['_sigma2']

    def predict(self, _X, return_mse=False):

        mX, sX, nX = self.model['mX'], self.model['sX'], self.model['nX']
        mY, sY = self.model['mY'], self.model['sY']
        regr, kernel, theta = self.regr, self.kernel, self.model["theta"]
        beta, gamma = self.model['beta'], self.model['gamma']

        # normalize the input given the mX and sX that was fitted before
        # NOTE: For the values to predict the _ is added to clarify its not the data fitted before
        _nX = (_X - mX) / sX

        # calculate regression and kernel
        _F = regr(_nX)
        _R = calc_kernel_matrix(_nX, nX, kernel, theta)

        # predict and destandardize
        _sY = _F @ beta + (gamma.T @ _R.T).T
        _Y = (_sY * sY) + mY

        ret = [_Y]

        if return_mse:
            rt = np.linalg.lstsq(self.model['C'], _R.T, rcond=None)[0]
            #np.linalg.lstsq(self.model["G"],((self.model["Ft"].T @ rt).T - _F))
            #u = ((self.model["Ft"].T @ rt).T - _F) / self.model["G"]
            u = ((self.model["Ft"].T @ rt).T - _F) @ np.linalg.inv(self.model["G"])
            mse = self.model["sigma2"] * (1 + np.sum(u ** 2, axis=1) - np.sum(rt ** 2, axis=0))
            ret.append(mse[:, None])

        if len(ret) == 1:
            return ret[0]
        else:
            return tuple(ret)

    def boxmin(self):

        itpar = start(self.theta, self)
        model = itpar["models"][-1]
        p, f = itpar["p"], model["f"]

        kmax = 2 if p <= 2 else min(p, 4)

        # if the initial guess is feasible
        if not np.isinf(f):

            for k in range(kmax):
                # save the last theta before exploring
                last_t = itpar["best"]["theta"]

                # do the actual explore step
                explore(self, itpar)
                move(last_t, self, itpar)

        self.itpar = itpar
