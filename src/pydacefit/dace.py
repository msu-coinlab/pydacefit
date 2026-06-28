"""DACE Kriging surrogate: fit, predict (with MSE/gradients), and theta optimization."""

from dataclasses import dataclass

import numpy as np

from pydacefit.corr import Gaussian, calc_kernel_matrix
from pydacefit.fit import DaceFitError, fit
from pydacefit.optimizers import LBFGS, Boxmin
from pydacefit.regr import ConstantRegression


@dataclass(frozen=True)
class Prediction:
    """The result of ``DACE.predict``: the mean plus any requested extras, named.

    ``y`` is always populated; ``mse``, ``grad`` and ``mse_grad`` are ``None`` unless the
    flags that produce them were set on the call. The type is the same every call, so
    callers read fields instead of guessing tuple positions, and a single ``predict``
    pass shares the kernel matrix and Cholesky solve across the mean and variance.

    Args:
        y: Predicted mean, shape ``(m, q)``.
        mse: Predictive variance (kriging MSE), shape ``(m, 1)``, or None when not requested.
            Shared across outputs for a multi-output model, so it stays ``(m, 1)``.
        grad: Gradient of the mean w.r.t. the query point, or None. ``(m, d)`` for a
            single-output model; ``(m, q, d)`` (one gradient per output) for multi-output.
        mse_grad: Gradient of the predictive variance w.r.t. the query point, ``(m, d)``, or None.
            Populated only when ``predict`` is called with both ``mse=True`` and ``grad=True``
            (it reuses the variance and mean-gradient terms). Lets a caller form ``grad(std)``
            as ``mse_grad / (2*sqrt(mse))`` -- e.g. for gradient-based Expected Improvement.
    """

    y: np.ndarray
    mse: np.ndarray | None = None
    grad: np.ndarray | None = None
    mse_grad: np.ndarray | None = None


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

        Args:
            regr: Regression trend instance: ConstantRegression(), LinearRegression() or QuadraticRegression().
                Defaults to ConstantRegression().
            corr: Correlation (kernel) instance, e.g. Gaussian(), Cubic(), Exponential(),
                RationalQuadratic(alpha=...). Defaults to Gaussian().
            theta: Initial value of theta. Can be a vector or a float
            thetaL: The lower bound if theta should be optimized.
            thetaU: The upper bound if theta should be optimized.
            optimizer: Strategy used to optimize theta when bounds are given (an instance of
                pydacefit.optimizers.Optimizer). Defaults to Boxmin() (pattern search);
                LBFGS() and Fixed() are the other built-ins.
            noise: Deliberate observation noise added to the diagonal of the correlation
                matrix on every fit. Because that diagonal is unit, it is a noise-to-signal
                ratio: noise=0.1 models 10% noise. 0.0 (default) interpolates the data;
                noise>0 makes a regression GP that smooths through the points instead.
            max_noise: Extra auto-repair budget added *on top of* ``noise`` to make an otherwise-
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

        Args:
            X: Training inputs, shape ``(n, d)``.
            Y: Training targets, shape ``(n,)`` or ``(n, q)`` for multi-output.
            validation: Optional binary mask over the rows of ``X`` (one entry per row), ``None``
                by default. A truthy entry marks a row as *held out for theta selection*:
                theta candidates are fit on the ``0`` rows and scored on the held-out rows
                (in normalized space), and the theta with the lowest held-out error is
                chosen instead of the maximum-likelihood one. ``None`` keeps the pure MLE
                behavior. Has no effect without theta bounds -- there is no search to steer.
            append: What the *final* model is fit on once theta is chosen, when a mask is given.
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

        Args:
            X: The new input points to add (only the additions, not the full set).
            Y: The target values corresponding to the new points ``X``.
            optimizer: Strategy to use for this refit only. ``None`` (default) uses ``LBFGS()`` --
                a fast, warm-started local refine, the natural choice for a refit since the
                optimum barely moves. Pass ``Boxmin()`` for a global re-search, or
                ``Fixed()`` to freeze theta and just re-solve. Has no effect when the model
                was built without theta bounds. Requires a prior successful ``fit``.
            validation: Whether the *new* points are held out to select theta. ``True`` (default)
                makes the new points the validation set (the existing data is the training
                set), so theta is chosen by how well the old data predicts the new points --
                a generalization-driven update. ``False`` re-fits by likelihood over all
                data. Either way the new points are appended (refit always appends).

        Raises:
            Exception: If called before any successful ``fit``.
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

        Args:
            model: A fit() result (carries beta, gamma, theta, kernel, regr), fit on the
                training rows.
            nXv: Held-out inputs, standardized, shape ``(m, d)``.
            nYv: Held-out targets, standardized, shape ``(m,)`` or ``(m, q)``.

        Returns:
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

        Args:
            _X: Query inputs, shape ``(m, d)``.
            mse: Also return the predictive variance (kriging MSE), shape ``(m, 1)``. For a
                multi-output model the variance is *shared* across outputs (the kernel and
                theta are shared), so it stays ``(m, 1)`` regardless of the number of outputs.
            grad: Also return the gradient of the mean w.r.t. the query point. Single-output
                models return ``(m, d)``; multi-output models return ``(m, q, d)`` -- one
                gradient per output. This is what a gradient-based optimizer searches over.

        Returns:
            ``y`` (always), plus ``mse`` / ``grad`` when their flag is set (else None).
            When ``mse`` and ``grad`` are *both* set, ``mse_grad`` (the variance gradient,
            ``(m, d)``) is also returned -- it shares the variance and mean-gradient terms,
            so the extra cost is one triangular solve per point.
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
        _mse_clamped = None  # mask of points whose variance was clamped (for mse_grad)
        if mse:
            rt = np.linalg.lstsq(self.model["C"], _R.T, rcond=None)[0]
            Ginv = np.linalg.inv(self.model["G"])
            u = (self.model["Ft"].T @ rt).T - _F
            v = u @ Ginv
            _mse = (self.model["sigma2"] * (1 + np.sum(v**2, axis=1) - np.sum(rt**2, axis=0)))[:, None]
            # the kriging variance is non-negative by theory; negatives are rounding near
            # training points (the cubic/spline kernels can dip moderately negative over a
            # region). Clamp at 0 so downstream sqrt(mse) (std, EI) never returns NaN.
            _mse_clamped = (_mse < 0.0).ravel()
            _mse = np.maximum(_mse, 0.0)

        # mse_grad is available alongside both the variance and the mean gradient, since
        # it reuses their terms (rt, v, Ginv and the batched dR/dF). sigma2 is already
        # destandardized and the dimensionless bracket is in normalized space, so the chain
        # rule to the original input is a single 1/sX scaling per dimension.
        want_mse_grad = mse and grad

        _grad = None
        _mse_grad = None
        if grad:
            # Fully vectorized over the m query points -- no per-point Python loop. dR is
            # the kernel gradient d r(x_i, t_j)/d x_i for every query/train pair from a
            # single corr.grad call; dF is the regression-basis Jacobian per query point.
            m, d = _nX.shape
            n = nX.shape[0]
            q = _sY.shape[1]
            d_all = np.repeat(_nX, n, axis=0) - np.tile(nX, (m, 1))  # (m*n, d)
            dR = corr.grad(d_all, theta).reshape(m, n, d)  # (m, n, d)
            dF = self.regr.grad(_nX)  # (m, d, p)

            # mean gradient (dF @ beta) + (gamma^T @ dR) per point -> (m, q, d), then
            # destandardize per output (sY) and per input dimension (1/sX). Single-output
            # keeps the historical (m, d) shape; multi-output is (m, q, d).
            mean_grad = np.einsum("idp,pq->iqd", dF, beta) + np.einsum("nq,ind->iqd", gamma, dR)
            mean_grad = mean_grad * sY[None, :, None] / sX[None, None, :]
            _grad = mean_grad[:, 0, :] if q == 1 else mean_grad

            if want_mse_grad:
                # variance gradient of the bracket 1 + ||v||^2 - ||rt||^2. One batched solve
                # gives drt = C^-1 dR for every point; du/dv reuse the mean's Ft, Ginv, rt, v.
                # C is the (square, full-rank) Cholesky factor, so a direct solve is exact
                # and ~75x faster than lstsq's SVD on the (n, m*d) right-hand side.
                b = dR.transpose(1, 0, 2).reshape(n, m * d)
                drt = np.linalg.solve(self.model["C"], b).reshape(n, m, d).transpose(1, 0, 2)
                du = np.einsum("np,ind->idp", self.model["Ft"], drt) - dF  # (m, d, p)
                dv = np.einsum("idp,pk->idk", du, Ginv)  # (m, d, p)
                d_bracket = 2.0 * (np.einsum("idp,ip->id", dv, v) - np.einsum("ind,ni->id", drt, rt))
                _mse_grad = self.model["sigma2"] * d_bracket / sX  # (m, d)
                # where the variance was clamped to 0 the clamped surface is flat -> 0 grad
                _mse_grad[_mse_clamped] = 0.0

        return Prediction(y=_Y, mse=_mse, grad=_grad, mse_grad=_mse_grad)
