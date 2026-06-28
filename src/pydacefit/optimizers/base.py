"""Base class and shared positive-definite feasibility helper for theta optimizers."""

from abc import ABC, abstractmethod

import numpy as np

from pydacefit.fit import DaceFitError, fit


class Optimizer(ABC):
    """Strategy that selects the correlation hyperparameter theta for a DACE fit.

    An optimizer is constructed with its own settings and then invoked by
    ``DACE.fit`` through ``optimize``. The held-out validation set is *not* part of an
    optimizer's construction -- it is data, so it is passed to ``optimize`` per fit.
    Subclasses implement the search and return the chosen model together with an
    optional trajectory for inspection. Implementations obtain their *committed* fits
    through ``fit_feasible`` so they honor the model's noise / ``max_noise`` policy
    consistently (the per-step search fits stay strict -- a non-PD theta is infeasible);
    the *searching* strategies pick their final model through ``_select`` so
    validation-based selection works uniformly (``Fixed`` does no search, so it has no
    trajectory to select from).
    """

    @abstractmethod
    def optimize(self, dace, validation=None):
        """Choose theta for the given (partially built) model.

        Parameters
        ----------
        dace : DACE
            Provides the standardized data in ``dace.model`` ("nX"/"nY"), the
            ``regr`` / ``kernel``, the start ``theta``, the bounds ``tl`` / ``tu``
            and the ``max_noise`` policy. When ``validation`` is given, ``dace.model``
            holds only the *training* rows; the held-out rows arrive via ``validation``.

        validation : tuple or None
            Either None (select theta by maximum likelihood -- the default) or a
            ``(nX_val, nY_val)`` tuple of the held-out rows, already standardized with
            the training stats. When given, theta is chosen to minimize prediction
            error on these rows instead of the likelihood: the search is still
            MLE-driven, but the final pick among the visited thetas is the one with the
            lowest held-out error. Scored in normalized space by ``DACE._val_error``.

        Returns
        -------
        tuple
            ``(model, itpar)`` -- the chosen fit() result and a search trajectory
            (a dict for Boxmin, or None when there is none).
        """

    def _select(self, dace, candidates, validation=None):
        """Pick the best feasible candidate fit from a search trajectory.

        With no validation set this is the maximum-likelihood choice -- the lowest
        objective ``f``, identical to the historical behavior. With a validation
        set it is the candidate with the lowest held-out RMSE: MLE drives the
        search, the validation set makes the final pick. Infeasible placeholders
        (which carry no built model) are skipped.

        Parameters
        ----------
        dace : DACE
            The model being fit; provides normalized-space scoring through
            ``_val_error`` against the training rows in ``dace.model``.

        candidates : list of dict
            The fit() results visited during the search.

        validation : tuple or None
            The held-out ``(nX_val, nY_val)`` rows (standardized), or None for the
            maximum-likelihood pick.

        Returns
        -------
        dict
            The selected fit() result.
        """
        feasible = [m for m in candidates if "gamma" in m]
        if not feasible:
            return candidates[-1]
        if validation is None:
            return min(feasible, key=lambda m: m["f"])
        nXv, nYv = validation
        return min(feasible, key=lambda m: dace._val_error(m, nXv, nYv))


def fit_feasible(dace, theta, relocate=True):
    """Fit at ``theta`` while guaranteeing a positive-definite correlation matrix.

    Two tools, in order of preference. Both fits carry the model's deliberate ``noise``.
    **Relocation** (when ``relocate`` is True and R is not PD at ``theta``): move theta up
    toward ``thetaU`` (R -> I, positive definite for the supported kernels) to find a
    feasible value -- this yields an exact fit at the deliberate noise (no climbing) and
    is strictly better than adding more noise when a feasible theta exists.
    **Noise climbing** (only if no theta in the bounds is feasible): fall back to a fit at
    the geometric-midpoint theta (or ``theta`` when not relocating) and climb the noise up
    to the model's ``max_noise`` ceiling. With ``max_noise=0`` no climbing is allowed, so
    this raises (strict); a positive ceiling yields the smallest positive-definite fit and
    records the noise, or raises if even ``max_noise`` is not enough.

    Parameters
    ----------
    dace : DACE
        Supplies the standardized data, kernel, regression, bounds, ``noise`` and
        ``max_noise``.

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
    DaceFitError
        If no positive-definite fit is possible within the ``max_noise`` ceiling.
    """
    X, Y = dace.model["nX"], dace.model["nY"]
    lo, up = dace.tl, dace.tu
    t_try = np.copy(theta) if isinstance(theta, np.ndarray) else theta

    # 1. preferred: an exact positive-definite fit at the deliberate noise (moving up
    #    toward thetaU when relocating). No climbing -- relocation finds a feasible theta.
    for _ in range(64):
        try:
            return t_try, fit(X, Y, dace.regr, dace.kernel, t_try, noise=dace.noise)
        except DaceFitError:
            if not relocate:
                break
            nxt = np.minimum(t_try * 2.0, up)
            if np.all(nxt == t_try):  # already at the upper bound -> nowhere left to move
                break
            t_try = nxt

    # 2. no feasible theta anywhere -> climb the noise at the fallback theta up to the
    #    model's max_noise ceiling. fit() climbs from `noise` to the smallest amount that
    #    works (recording it) or raises if even max_noise is not enough; max_noise=0 means
    #    no climbing, so this is the strict raise.
    if relocate:
        t_fb = np.sqrt(np.atleast_1d(np.asarray(lo, dtype=float)) * np.atleast_1d(np.asarray(up, dtype=float)))
    else:
        t_fb = theta
    try:
        return t_fb, fit(X, Y, dace.regr, dace.kernel, t_fb, noise=dace.noise, max_noise=dace.max_noise)
    except DaceFitError as e:
        raise DaceFitError(
            "No positive-definite correlation matrix for theta in [thetaL, thetaU]; "
            "increase max_noise to allow a noise-regularized fallback fit."
        ) from e
