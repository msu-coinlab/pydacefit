"""DACE Kriging surrogate: fit, predict (with MSE/gradients), and theta optimization."""

from dataclasses import dataclass

import numpy as np

from pydacefit.corr import Gaussian, calc_grad, calc_kernel_matrix
from pydacefit.fit import DaceFitError, fit
from pydacefit.optimizers import LBFGS, Boxmin
from pydacefit.regr import ConstantRegression


@dataclass(frozen=True)
class Prediction:
    """The result of ``DACE.predict``: the mean plus any requested extras, named.

    ``y`` is always populated; ``mse`` and ``grad`` are ``None`` unless their flag was
    set on the call. The type is the same every call, so callers read fields instead of
    guessing tuple positions, and a single ``predict`` pass shares the kernel matrix and
    Cholesky solve across the mean and variance.

    Parameters
    ----------
    y : numpy.ndarray
        Predicted mean, shape ``(m, q)``.

    mse : numpy.ndarray or None
        Predictive variance (kriging MSE), shape ``(m, 1)``, or None when not requested.

    grad : numpy.ndarray or None
        Gradient of the mean w.r.t. the query point, ``(m, d)``, or None.
    """

    y: np.ndarray
    mse: np.ndarray | None = None
    grad: np.ndarray | None = None


class DACE:
    def __init__(
        self,
        regr=None,
        corr=None,
        theta=1.0,
        thetaL=0.0,
        thetaU=100.0,
        optimizer=None,
        noise=0.0,
        max_noise=1e-4,
    ):
        """Construct the model with the given regression and correlation types.

        It can be initialized with different regression and correlation types, and
        whether hyperparameter optimization is used is controlled by the theta bounds.

        Parameters
        ----------
        regr : Regression, optional
            Regression trend instance: ConstantRegression(), LinearRegression() or QuadraticRegression().
            Defaults to ConstantRegression().

        corr : Correlation, optional
            Correlation (kernel) instance, e.g. Gaussian(), Cubic(), Exponential(),
            RationalQuadratic(alpha=...). Defaults to Gaussian().

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

        noise : float
            Deliberate observation noise added to the diagonal of the correlation
            matrix on every fit. Because that diagonal is unit, it is a noise-to-signal
            ratio: noise=0.1 models 10% noise. 0.0 (default) interpolates the data;
            noise>0 makes a regression GP that smooths through the points instead.

        max_noise : float
            Extra auto-repair budget added *on top of* ``noise`` to make an otherwise-
            infeasible committed fit possible. When the correlation matrix is not
            positive-definite at ``noise``, a separate repair term climbs from a tiny
            floor up to ``max_noise`` (smallest amount that works) and the total effective
            diagonal is recorded in model["noise"], with a warning. Being independent of
            ``noise`` means the budget works no matter how much deliberate noise was set.
            The default ``1e-4`` auto-repairs tiny numerical non-positive-definiteness
            (near-duplicate points, mild ill-conditioning) with negligible bias; genuine
            conditional non-positive-definiteness (e.g. cubic at a bad theta) needs far
            more and still raises. Set ``0.0`` for strict interpolation (raise on any
            non-PD). The theta *search* never repairs -- a non-PD theta is simply
            infeasible.
        """
        super().__init__()
        self.regr = regr if regr is not None else ConstantRegression()
        self.kernel = corr if corr is not None else Gaussian()

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

        # diagonal noise-to-signal terms: `noise` is deliberate (always added; >0 -> a
        # regression GP), `max_noise` is the ceiling the noise is climbed to when a fit
        # is otherwise infeasible. Both 0.0 -> strict interpolation, raise on non-PD.
        self.noise = noise
        self.max_noise = max_noise

        # record of the hyperparameter optimization (search trajectory + diagnostics),
        # populated by the optimizer on fit; None until then / for a fixed theta.
        self.optimization = None

    def fit(self, X, Y, validation=None, append=True):
        """Fit the model, optionally selecting theta on a held-out subset of the rows.

        Parameters
        ----------
        X : numpy.ndarray
            Training inputs, shape ``(n, d)``.

        Y : numpy.ndarray
            Training targets, shape ``(n,)`` or ``(n, q)`` for multi-output.

        validation : numpy.ndarray or None
            Optional binary mask over the rows of ``X`` (one entry per row), ``None``
            by default. A truthy entry marks a row as *held out for theta selection*:
            theta candidates are fit on the ``0`` rows and scored on the held-out rows
            (in normalized space), and the theta with the lowest held-out error is
            chosen instead of the maximum-likelihood one. ``None`` keeps the pure MLE
            behavior. Has no effect without theta bounds -- there is no search to steer.

        append : bool
            What the *final* model is fit on once theta is chosen, when a mask is given.
            ``True`` (default) refits on all rows, so the held-out rows rejoin and
            ``predict`` uses every label. ``False`` keeps the model fit on the ``0``
            rows only, so it never saw the held-out rows (useful when their error is
            reported separately). Ignored when ``validation`` is ``None``.
        """
        # the targets should be a 2d array
        if len(Y.shape) == 1:
            Y = Y[:, None]

        # check if for each observation a target values exist
        if X.shape[0] != Y.shape[0]:
            raise Exception("X and Y must have the same number of rows.")

        # save the mean and standard deviation of the input. Stats are over all rows,
        # so the held-out validation rows share the training normalization -- selection
        # then scores in this one normalized space (no destandardization, scale-free).
        mX, sX = np.mean(X, axis=0), np.std(X, axis=0, ddof=1)
        mY, sY = np.mean(Y, axis=0), np.std(Y, axis=0, ddof=1)

        # guard zero-variance columns/outputs (a constant degenerates the normalization,
        # dividing by zero and poisoning the fit with NaN). Setting std to 1 maps the
        # constant to all-zeros after centering, so the fit stays finite and a constant
        # target degrades gracefully to a constant predictor: predict -> its mean, since
        # destandardizing 0 gives 0*sY + mY = mY.
        sX = np.where(sX == 0.0, 1.0, sX)
        sY = np.where(sY == 0.0, 1.0, sY)

        # standardize the input
        nX = (X - mX) / sX
        nY = (Y - mY) / sY

        stats = {"mX": mX, "sX": sX, "mY": mY, "sY": sY}
        optimize_theta = self.tl is not None and self.tu is not None

        if optimize_theta and validation is not None:
            # held-out theta selection: train candidates on the 0-rows and score them on
            # the held-out rows. The optimizer still searches by likelihood; only its
            # final pick uses the held-out set (passed already normalized, see
            # _val_error). dace.model["nX"] is the training split during the search.
            mask = np.asarray(validation, dtype=bool)
            if mask.shape[0] != X.shape[0]:
                raise Exception("validation mask must have one entry per row of X.")
            n_held = int(mask.sum())
            if n_held == 0 or n_held == mask.shape[0]:
                raise Exception(
                    f"validation mask must hold out some rows but not all (got {n_held} of {mask.shape[0]})."
                )
            train = ~mask
            self.model = {"nX": nX[train], "nY": nY[train], **stats}
            selected, self.optimization = self.optimizer.optimize(self, validation=(nX[mask], nY[mask]))

            if append:
                # theta is chosen -> refit the final model on ALL rows so predict uses
                # every label (honoring the noise / max_noise policy, like any committed fit).
                # The theta was positive-definite on the training rows; re-adding the
                # held-out rows can in principle break that, so surface a clear,
                # actionable error instead of a bare Cholesky failure. self.optimization
                # keeps the train-only search trajectory for inspection.
                try:
                    self.model = fit(
                        nX, nY, self.regr, self.kernel, selected["theta"], noise=self.noise, max_noise=self.max_noise
                    )
                except DaceFitError as e:
                    raise DaceFitError(
                        "The validation-selected theta is not positive-definite once the held-out rows "
                        "rejoin the model (append=True). Raise max_noise to regularize the final fit, "
                        "or pass append=False to keep the train-only model."
                    ) from e
                Xf, Yf, nXf, nYf = X, Y, nX, nY
            else:
                # keep the train-only model; it never saw the held-out rows
                self.model = selected
                Xf, Yf, nXf, nYf = X[train], Y[train], nX[train], nY[train]

        elif optimize_theta:
            self.model = {"nX": nX, "nY": nY, **stats}
            self.model, self.optimization = self.optimizer.optimize(self)
            Xf, Yf, nXf, nYf = X, Y, nX, nY

        else:
            self.model = fit(nX, nY, self.regr, self.kernel, self.theta, noise=self.noise, max_noise=self.max_noise)
            self.optimization = None
            Xf, Yf, nXf, nYf = X, Y, nX, nY

        # keep the raw (destandardized) training data so refit() can append to it
        self.model = {**self.model, "X": Xf, "Y": Yf, **stats, "nX": nXf, "nY": nYf}
        self.model["sigma2"] = np.square(sY) @ self.model["_sigma2"]

    def refit(self, X, Y, optimizer=None, validation=True):
        """Append new observations to the training data and re-fit, warm-started.

        Takes only the *new* points, appends them to the data the model was last
        fit on, and re-fits on the combined set. Theta is seeded from the previous
        fit (warm start), so the search begins next to the optimum instead of at
        the original initial guess -- the optimum barely moves when a few points
        are added, which is exactly when a local optimizer shines. The new points
        are always appended to the model (that is what refit means); ``validation``
        only controls whether they also *steer* the theta search.

        The defaults make refit cheap and self-tuning: a warm-started ``LBFGS()`` with
        no restarts (a handful of gradient steps from the previous optimum) and the new
        points as the held-out set -- so each refit nudges theta toward whatever best
        predicts the freshly added points, exactly what an online/surrogate loop wants.

        Parameters
        ----------
        X : numpy.ndarray
            The new input points to add (only the additions, not the full set).

        Y : numpy.ndarray
            The target values corresponding to the new points ``X``.

        optimizer : Optimizer, optional
            Strategy to use for this refit only. ``None`` (default) uses ``LBFGS()`` --
            a fast, warm-started local refine, the natural choice for a refit since the
            optimum barely moves. Pass ``Boxmin()`` for a global re-search, or
            ``Fixed()`` to freeze theta and just re-solve. Has no effect when the model
            was built without theta bounds. Requires a prior successful ``fit``.

        validation : bool
            Whether the *new* points are held out to select theta. ``True`` (default)
            makes the new points the validation set (the existing data is the training
            set), so theta is chosen by how well the old data predicts the new points --
            a generalization-driven update. ``False`` re-fits by likelihood over all
            data. Either way the new points are appended (refit always appends).

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
        n_old = self.model["X"].shape[0]
        X = np.vstack([self.model["X"], X])
        Y = np.vstack([self.model["Y"], Y])

        # warm start: seed the search with the previously optimized theta
        self.theta = self.model["theta"]

        # if requested, the appended rows are the held-out set that selects theta
        mask = None
        if validation:
            mask = np.zeros(X.shape[0], dtype=bool)
            mask[n_old:] = True

        # per-refit optimizer, restored afterwards. Default is a warm-started LBFGS (no
        # restarts) -- a fast local refine -- not the model's configured optimizer,
        # which is meant for the cold initial fit.
        configured = self.optimizer
        self.optimizer = optimizer or LBFGS()
        try:
            self.fit(X, Y, validation=mask)
        finally:
            self.optimizer = configured

    def _val_error(self, model, nXv, nYv):
        """Root-mean-square error of a candidate fit on the held-out rows.

        Used by an optimizer to select theta when a validation mask is given. The
        candidate was fit on the training rows; here it predicts the held-out rows and
        the error is measured in *normalized* space. The held-out inputs and targets
        arrive already standardized with the training stats (the mask is applied to the
        same ``nX`` / ``nY`` the candidate trained on), so there is nothing to
        destandardize and the criterion is scale-free across outputs.

        Parameters
        ----------
        model : dict
            A fit() result (carries beta, gamma, theta, kernel, regr), fit on the
            training rows.

        nXv : numpy.ndarray
            Held-out inputs, standardized, shape ``(m, d)``.

        nYv : numpy.ndarray
            Held-out targets, standardized, shape ``(m,)`` or ``(m, q)``.

        Returns
        -------
        float
            The RMSE in normalized Y space.
        """
        nX = self.model["nX"]  # the training rows the candidate was fit on

        _F = model["regr"](nXv)
        _R = calc_kernel_matrix(nXv, nX, model["kernel"], model["theta"])

        # predicted normalized targets, compared directly to the normalized held-out Y
        _sYhat = _F @ model["beta"] + (model["gamma"].T @ _R.T).T

        nYv = nYv[:, None] if nYv.ndim == 1 else nYv
        return float(np.sqrt(np.mean(np.square(_sYhat - nYv))))

    def predict(self, _X, mse=False, grad=False):
        """Predict the mean, optionally the variance and the mean's gradient, in one pass.

        Mean and variance share the kernel matrix and the Cholesky solve, so computing
        them together is cheaper than two calls -- this is why both live on one method.

        Parameters
        ----------
        _X : numpy.ndarray
            Query inputs, shape ``(m, d)``.

        mse : bool
            Also return the predictive variance (kriging MSE), shape ``(m, 1)``.

        grad : bool
            Also return the gradient of the mean w.r.t. the query point, ``(m, d)`` --
            what a gradient-based optimizer uses to search over the surrogate.

        Returns
        -------
        Prediction
            ``y`` (always), plus ``mse`` / ``grad`` when their flag is set (else None).
        """
        mX, sX, nX = self.model["mX"], self.model["sX"], self.model["nX"]
        mY, sY = self.model["mY"], self.model["sY"]
        regr, corr, theta = self.regr, self.kernel, self.model["theta"]
        beta, gamma = self.model["beta"], self.model["gamma"]

        # normalize the input given the mX and sX that was fitted before
        # NOTE: For the values to predict the _ is added to clarify its not the data fitted before
        _nX = (_X - mX) / sX

        # calculate regression and corr (the kernel matrix shared by mean and mse)
        _F = regr(_nX)
        _R = calc_kernel_matrix(_nX, nX, corr, theta)

        # predict and destandardize
        _sY = _F @ beta + (gamma.T @ _R.T).T
        _Y = (_sY * sY) + mY

        _mse = None
        if mse:
            rt = np.linalg.lstsq(self.model["C"], _R.T, rcond=None)[0]
            u = (self.model["Ft"].T @ rt).T - _F
            v = u @ np.linalg.inv(self.model["G"])
            _mse = (self.model["sigma2"] * (1 + np.sum(v**2, axis=1) - np.sum(rt**2, axis=0)))[:, None]

        _grad = None
        if grad:
            # the gradient must be calculated for each point at once
            _grad = np.zeros(_X.shape)
            for i, _x in enumerate(_nX):
                _dF = self.regr.grad(_x[None, :])
                _dR = calc_grad(_x[None, :], nX, corr.grad, theta)
                dy = (_dF @ beta).T + gamma.T @ _dR
                _grad[i] = dy * sY / sX

        return Prediction(y=_Y, mse=_mse, grad=_grad)
