"""DACE Kriging surrogate: fit, predict (with MSE/gradients), and theta optimization."""

import numpy as np

from pydacefit import corr as _corr
from pydacefit import regr as _regr
from pydacefit.corr import calc_grad, calc_kernel_matrix, corr_gauss
from pydacefit.fit import fit
from pydacefit.optimizers import Boxmin
from pydacefit.regr import regr_constant


class DACE:
    def __init__(
        self,
        regr=regr_constant,
        corr=corr_gauss,
        theta=1.0,
        thetaL=0.0,
        thetaU=100.0,
        optimizer=None,
        raise_error=True,
    ):
        """Construct the model with the given regression and correlation types.

        It can be initialized with different regression and correlation types, and
        whether hyperparameter optimization is used is controlled by the theta bounds.

        Parameters
        ----------
        regr : callable
            Type of regression that should be used: regr_constant, regr_linear or regr_quadratic

        corr : callable
            Type of correlation (kernel) that should be used. default: corr_gauss

        theta : float
            Initial value of theta. Can be a vector or a float

        thetaL : float
            The lower bound if theta should be optimized.

        thetaU : float
            The upper bound if theta should be optimized.

        optimizer : Optimizer, optional
            Strategy used to optimize theta when bounds are given (an instance of
            pydacefit.optimizers.Optimizer). Defaults to Boxmin() (pattern search);
            LBFGS() and Fixed() are the other built-ins.

        raise_error : bool
            What to do when no theta in [thetaL, thetaU] yields a positive-definite
            correlation matrix. True (default) raises; False falls back to a
            regularized model at the geometric-midpoint theta (with a warning).
        """
        super().__init__()
        self.regr = regr
        self.kernel = corr

        # most of the model will be stored here
        self.model = None

        # the hyperparameter can be defined (coerce a list like the bounds below, so
        # a vector theta reaches the kernel as an array and not a Python list)
        self.theta = np.array(theta) if type(theta) is list else theta

        # lower and upper bound if it should be optimized
        self.tl = np.array(thetaL) if type(thetaL) is list else thetaL
        self.tu = np.array(thetaU) if type(thetaU) is list else thetaU

        # strategy that optimizes theta within the bounds (see pydacefit.optimizers)
        self.optimizer = optimizer if optimizer is not None else Boxmin()

        # whether to raise (vs. fall back to a regularized model) when no theta is feasible
        self.raise_error = raise_error

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
            self.model = {"nX": nX, "nY": nY}
            self.model, self.itpar = self.optimizer.optimize(self)
        else:
            self.model = fit(nX, nY, self.regr, self.kernel, self.theta)
            self.itpar = None

        # keep the raw (destandardized) training data so refit() can append to it
        self.model = {**self.model, "X": X, "Y": Y, "mX": mX, "sX": sX, "mY": mY, "sY": sY, "nX": nX, "nY": nY}
        self.model["sigma2"] = np.square(sY) @ self.model["_sigma2"]

    def refit(self, X, Y, optimizer=None):
        """Append new observations to the training data and re-fit, warm-started.

        Takes only the *new* points, appends them to the data the model was last
        fit on, and re-fits on the combined set. Theta is seeded from the previous
        fit (warm start), so the search begins next to the optimum instead of at
        the original initial guess -- the optimum barely moves when a few points
        are added, which is exactly when a local optimizer shines.

        Parameters
        ----------
        X : numpy.ndarray
            The new input points to add (only the additions, not the full set).

        Y : numpy.ndarray
            The target values corresponding to the new points ``X``.

        optimizer : Optimizer, optional
            Strategy to use for this refit only. ``None`` (default) inherits the
            model's configured optimizer. Pass ``LBFGS()`` for a fast local refine,
            or ``Fixed()`` to freeze theta and just re-solve. Has no effect when the
            model was built without theta bounds. Requires a prior successful ``fit``.

        Raises
        ------
        Exception
            If called before any successful ``fit``.
        """
        if self.model is None:
            raise Exception("refit() requires a prior fit(); call fit() first.")

        # match fit's reshape so a 1d Y appends cleanly onto the stored 2d targets
        if len(Y.shape) == 1:
            Y = Y[:, None]

        # append the new observations to the data the model was last fit on
        X = np.vstack([self.model["X"], X])
        Y = np.vstack([self.model["Y"], Y])

        # warm start: seed the search with the previously optimized theta
        self.theta = self.model["theta"]

        # optional per-refit optimizer override, restored afterwards
        configured = self.optimizer
        self.optimizer = optimizer or configured
        try:
            self.fit(X, Y)
        finally:
            self.optimizer = configured

    def predict(self, _X, return_mse=False, return_gradient=False, return_mse_gradient=False):

        mX, sX, nX = self.model["mX"], self.model["sX"], self.model["nX"]
        mY, sY = self.model["mY"], self.model["sY"]
        regr, corr, theta = self.regr, self.kernel, self.model["theta"]
        beta, gamma = self.model["beta"], self.model["gamma"]

        # normalize the input given the mX and sX that was fitted before
        # NOTE: For the values to predict the _ is added to clarify its not the data fitted before
        _nX = (_X - mX) / sX

        # calculate regression and corr
        _F = regr(_nX)
        _R = calc_kernel_matrix(_nX, nX, corr, theta)

        # predict and destandardize
        _sY = _F @ beta + (gamma.T @ _R.T).T
        _Y = (_sY * sY) + mY

        ret = [_Y]

        if return_mse:
            rt = np.linalg.lstsq(self.model["C"], _R.T, rcond=None)[0]
            u = (self.model["Ft"].T @ rt).T - _F
            v = u @ np.linalg.inv(self.model["G"])
            mse = self.model["sigma2"] * (1 + np.sum(v**2, axis=1) - np.sum(rt**2, axis=0))
            ret.append(mse[:, None])

        if return_gradient:
            # the final gradient matrix
            _grad = np.zeros(_X.shape)

            # the gradient must be calculated for each point at once
            for i, _x in enumerate(_nX):
                _dF = get_gradient_func(self.regr)(_x[None, :])
                _dR = calc_grad(_x[None, :], nX, get_gradient_func(corr), theta)

                dy = (_dF @ self.model["beta"]).T + self.model["gamma"].T @ _dR
                _grad[i] = dy * self.model["sY"] / self.model["sX"]

            ret.append(_grad)

        if return_mse_gradient:
            if not return_mse or not return_gradient:
                raise Exception("To evaluate the gradient of MSE, you must calculate the gradient and MSE as well.")

            # the final gradient matrix
            _mse_grad = np.zeros(_X.shape)

            # the gradient must be calculated for each point at once
            for i, _x in enumerate(_nX):
                # not implemented yet - precision problems occurred and results did not match
                _mse_grad[i] = np.nan

            ret.append(_mse_grad)

        if len(ret) == 1:
            return ret[0]
        else:
            return tuple(ret)


def get_gradient_func(func):
    # callable kernels may carry their own analytic gradient as a `.grad`;
    # plain-function kernels expose it as a module-level `<name>_grad`.
    own = getattr(func, "grad", None)
    if callable(own):
        return own
    name = getattr(func, "__name__", "") + "_grad"
    for module in (_corr, _regr):
        grad = getattr(module, name, None)
        if grad is not None:
            return grad
    return None
