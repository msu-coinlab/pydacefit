"""Base class and shared positive-definite feasibility helper for theta optimizers."""

import warnings
from abc import ABC, abstractmethod

import numpy as np

from pydacefit.fit import fit


class Optimizer(ABC):
    """Strategy that selects the correlation hyperparameter theta for a DACE fit.

    An optimizer is constructed with its own settings and then invoked by
    ``DACE.fit`` through ``optimize``. Subclasses implement the search and return the
    chosen model together with an optional trajectory for inspection. Implementations
    should obtain their fits through ``fit_feasible`` so they all honor the model's
    ``raise_error`` policy consistently, and pick their final model through
    ``_select`` so validation-based selection works uniformly across strategies.
    """

    def __init__(self, validation=None):
        """Store the optional held-out validation set used to select theta.

        Parameters
        ----------
        validation : tuple or None
            Either None (select theta by maximum likelihood -- the default) or a
            raw ``(X_val, Y_val)`` tuple in the same units as the training data.
            When given, theta is chosen to minimize prediction error on this set
            instead of the likelihood: the search is still MLE-driven, but the
            final pick among the visited thetas is the one with the lowest
            held-out error. The set is scored in original Y space by
            ``DACE._val_error``, so normalization stays inside the model.
        """
        self.validation = validation

    @abstractmethod
    def optimize(self, dace):
        """Choose theta for the given (partially built) model.

        Parameters
        ----------
        dace : DACE
            Provides the standardized data in ``dace.model`` ("nX"/"nY"), the
            ``regr`` / ``kernel``, the start ``theta``, the bounds ``tl`` / ``tu``
            and the ``raise_error`` policy.

        Returns
        -------
        tuple
            ``(model, itpar)`` -- the chosen fit() result and a search trajectory
            (a dict for Boxmin, or None when there is none).
        """

    def _select(self, dace, candidates):
        """Pick the best feasible candidate fit from a search trajectory.

        With no validation set this is the maximum-likelihood choice -- the lowest
        objective ``f``, identical to the historical behavior. With a validation
        set it is the candidate with the lowest held-out RMSE: MLE drives the
        search, the validation set makes the final pick. Infeasible placeholders
        (which carry no built model) are skipped.

        Parameters
        ----------
        dace : DACE
            The model being fit; provides normalization-aware scoring through
            ``_val_error`` and the training standardization.

        candidates : list of dict
            The fit() results visited during the search.

        Returns
        -------
        dict
            The selected fit() result.
        """
        feasible = [m for m in candidates if "gamma" in m]
        if not feasible:
            return candidates[-1]
        if self.validation is None:
            return min(feasible, key=lambda m: m["f"])
        Xv, Yv = self.validation
        return min(feasible, key=lambda m: dace._val_error(m, Xv, Yv))


def fit_feasible(dace, theta, relocate=True):
    """Fit at ``theta`` while guaranteeing a positive-definite correlation matrix.

    A non-positive-definite R makes the likelihood objective undefined, so such a
    theta is treated as infeasible -- a constraint -- rather than patched with a
    nugget. If ``relocate`` is True and R is not PD at ``theta``, the search moves
    theta up toward ``thetaU`` (R -> I, positive definite for the supported kernels)
    to find a feasible value. If nothing feasible is found, behavior follows
    ``dace.raise_error``: True raises; False falls back to a nugget-regularized model
    -- at the geometric-midpoint theta when relocating, otherwise at ``theta`` -- and
    warns.

    Parameters
    ----------
    dace : DACE
        Supplies the standardized data, kernel, regression and bounds.

    theta : float or numpy.ndarray
        The theta to fit at (already within the bounds).

    relocate : bool
        Whether to search upward for a feasible theta (True for the searching
        optimizers) or stay at ``theta`` (False for a fixed-theta fit).

    Returns
    -------
    tuple
        ``(theta_used, model)``.

    Raises
    ------
    Exception
        If no positive-definite fit is possible and ``dace.raise_error`` is True, or
        even regularization cannot make R positive definite.
    """
    X, Y = dace.model["nX"], dace.model["nY"]
    lo, up = dace.tl, dace.tu
    t_try = np.copy(theta) if isinstance(theta, np.ndarray) else theta

    # 1. preferred: a positive-definite fit (moving up toward thetaU when relocating)
    for _ in range(64):
        try:
            return t_try, fit(X, Y, dace.regr, dace.kernel, t_try)
        except Exception:
            if not relocate:
                break
            nxt = np.minimum(t_try * 2.0, up)
            if np.all(nxt == t_try):  # already at the upper bound -> nowhere left to move
                break
            t_try = nxt

    # 2. nothing feasible -> honor the user's choice
    if getattr(dace, "raise_error", True):
        raise Exception(
            "No positive-definite correlation matrix for theta in [thetaL, thetaU]; "
            "pass raise_error=False to fall back to a regularized model instead."
        )

    # 3. graceful fallback: regularize until R is positive definite
    if relocate:
        t_fb = np.sqrt(np.atleast_1d(np.asarray(lo, dtype=float)) * np.atleast_1d(np.asarray(up, dtype=float)))
    else:
        t_fb = theta
    for k in range(-12, 6):
        nugget = 10.0**k
        try:
            model = fit(X, Y, dace.regr, dace.kernel, t_fb, nugget=nugget)
        except Exception:
            continue
        warnings.warn(
            f"No positive-definite theta found; falling back with nugget={nugget:g} "
            f"(regularized model, not an exact interpolator).",
            stacklevel=2,
        )
        return t_fb, model

    raise Exception("Could not fit even with regularization; the data may be degenerate.")
