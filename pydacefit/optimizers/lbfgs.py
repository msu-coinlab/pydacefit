"""L-BFGS-B optimizer with an analytic theta-gradient (exact for the gaussian kernel)."""

import numpy as np
from scipy.optimize import minimize  # type: ignore[import-untyped]

from pydacefit import corr as _corr
from pydacefit.fit import fit
from pydacefit.optimizers.base import Optimizer, fit_feasible

# returned by a failed fit so the gradient-based optimizer sees a large but finite
# penalty (np.inf breaks finite-difference gradients) and steps away from it.
_INFEASIBLE = 1e25


def theta_grad_func(kernel):
    """Look up a kernel's analytic theta-gradient, or None if it has none.

    Mirrors ``dace.get_gradient_func``: callable kernels may carry their own
    ``theta_grad``; plain-function kernels expose it as ``<name>_theta_grad`` in the
    corr module (e.g. ``corr_gauss`` -> ``corr_gauss_theta_grad``).

    Parameters
    ----------
    kernel : callable
        The correlation function configured on the model.

    Returns
    -------
    callable or None
        The (D, theta) -> per-pair, per-theta derivative function, or None.
    """
    own = getattr(kernel, "theta_grad", None)
    if callable(own):
        return own
    return getattr(_corr, getattr(kernel, "__name__", "") + "_theta_grad", None)


def objective_gradient(nX, model, theta, grad_func):
    """Analytic gradient of the DACE objective ``f`` with respect to theta.

    Implements ``df/d(theta_k) = (D/n) * (S * tr(Rinv @ Rk) - sum_j g_j^T Rk g_j)``
    where ``D = det(R)**(1/n)``, ``S = sum_j sigma2_j``, ``g`` is the model's gamma,
    and ``Rk = dR/d(theta_k)`` comes from ``grad_func``. The cross term in dsigma2
    vanishes by the optimality of the GLS coefficients (envelope theorem).

    Parameters
    ----------
    nX : numpy.ndarray
        The standardized design sites the model was fit on.

    model : dict
        A fit() result (provides R, C, gamma and the per-output sigma2).

    theta : numpy.ndarray
        The theta at which the model was fit.

    grad_func : callable
        Kernel theta-derivative: ``(D, theta) -> (n_pairs, p)`` array.

    Returns
    -------
    numpy.ndarray
        The gradient, shape ``(p,)`` matching the optimized theta dimension.
    """
    R, C, gamma = model["R"], model["C"], model["gamma"]
    n = nX.shape[0]

    detR = float(np.prod(np.diag(C) ** (2.0 / n)))
    s = float(np.sum(model["_sigma2"]))
    Rinv = np.linalg.inv(R)

    # pairwise differences, same layout as calc_kernel_matrix -> reshape to (n, n)
    D = np.repeat(nX, n, axis=0) - np.tile(nX, (n, 1))
    dK = grad_func(D, theta)

    grad = np.zeros(dK.shape[1])
    for k in range(dK.shape[1]):
        Rk = dK[:, k].reshape(n, n)
        tr_term = np.sum(Rinv * Rk)  # tr(Rinv @ Rk), both symmetric
        quad_term = np.sum(gamma * (Rk @ gamma))  # sum_j g_j^T Rk g_j
        grad[k] = (detR / n) * (s * tr_term - quad_term)
    return grad


class LBFGS(Optimizer):
    """Bounded quasi-Newton (L-BFGS-B), parametrized by its stop tolerances.

    A *local* optimizer, the natural choice for ``refit``: from a warm start it
    converges in a few steps. When the kernel exposes an analytic theta-gradient
    (see ``theta_grad_func``, available for ``corr_gauss``) it is used as the exact
    Jacobian -- this is what makes it fast; otherwise it falls back to scipy's
    finite-difference gradient.

    The defaults are deliberately *relaxed*: a warm-started refit already sits next
    to the optimum, so chasing the last digits costs iterations without meaningfully
    changing the model. Loosen further (``LBFGS(gtol=1e-2)``) to save more, or
    tighten (``LBFGS(gtol=1e-8)``) when an exact optimum matters.

    The DACE likelihood is multi-modal, so a single local search can settle in a
    poor basin when started far from the optimum. ``n_restarts`` adds that many
    random log-uniform restarts within the bounds and keeps the best result. The
    first start is always the model's own (warm) theta, so ``n_restarts=0`` keeps
    the warm-start behavior unchanged -- restarts are only useful for cold fits from
    an unknown scale, where the configured start may miss the global optimum.

    Parameters
    ----------
    gtol : float
        Stop once the max-norm of the projected gradient falls below this.

    ftol : float
        Stop once the relative objective improvement falls below this.

    n_restarts : int
        Number of additional random restarts (beyond the configured start) to guard
        against local optima. 0 (default) means a single start from the model theta.

    random_state : int or None
        Seed for the restart sampling, so multi-start fits are reproducible.

    validation : tuple or None
        Optional ``(X_val, Y_val)`` held-out set (see ``Optimizer``). When given,
        theta is chosen by ranking the *entire* search history -- every theta the
        gradient search evaluates is recorded (at no extra cost, since the objective
        already fits a model there) and the one with the lowest held-out error is
        kept. So selection is meaningful even with ``n_restarts=0``, where a single
        descent still visits many thetas. The MLE path is unaffected: with no
        validation set the best per-restart optimum is returned, exactly as before.
    """

    def __init__(self, gtol=1e-3, ftol=1e-6, n_restarts=0, random_state=0, validation=None):
        super().__init__(validation=validation)
        self.gtol = gtol
        self.ftol = ftol
        self.n_restarts = n_restarts
        self.random_state = random_state

    def optimize(self, dace):
        """Run L-BFGS-B from the (warm) start plus any restarts; return ``(best_model, None)``."""
        nX, nY = dace.model["nX"], dace.model["nY"]
        regr, kernel = dace.regr, dace.kernel
        grad_func = theta_grad_func(kernel)
        options = {"gtol": self.gtol, "ftol": self.ftol}

        # bring theta and bounds to a common 1d shape (broadcast scalar bounds)
        theta0 = np.atleast_1d(np.array(dace.theta, dtype=float))
        lo = np.broadcast_to(np.atleast_1d(np.asarray(dace.tl, dtype=float)), theta0.shape)
        up = np.broadcast_to(np.atleast_1d(np.asarray(dace.tu, dtype=float)), theta0.shape)
        theta0 = np.clip(theta0, lo, up)
        bounds = list(zip(lo, up))

        # for validation selection we rank every theta the search visits, so record
        # each evaluated feasible model. fun already fits at every theta -- recording
        # is just keeping that model instead of discarding it (zero extra fits). For
        # the MLE path we never look at the history, so don't pay the memory.
        record = self.validation is not None
        history = []

        # objective (+ analytic gradient when the kernel provides one)
        if grad_func is not None:

            def fun(t):
                t = np.array(t, dtype=float)
                try:
                    model = fit(nX, nY, regr, kernel, t)
                except Exception:
                    return _INFEASIBLE, np.zeros_like(t)
                if record:
                    history.append(model)
                return float(model["f"]), objective_gradient(nX, model, t, grad_func)

            jac = True
        else:

            def fun(t):
                try:
                    model = fit(nX, nY, regr, kernel, np.array(t, dtype=float))
                except Exception:
                    return _INFEASIBLE
                if record:
                    history.append(model)
                return float(model["f"])

            jac = False

        # the first start is always the (warm) configured theta; the rest are random
        starts = [theta0]
        if self.n_restarts > 0:
            rng = np.random.default_rng(self.random_state)
            for _ in range(self.n_restarts):
                starts.append(10.0 ** rng.uniform(np.log10(lo), np.log10(up)))

        # build a model at each restart's converged theta (fit_feasible also applies
        # the shared feasibility / raise_error safety net). For n_restarts=0 this is a
        # single start, as before.
        optima = []
        for s in starts:
            res = minimize(fun, s, method="L-BFGS-B", jac=jac, bounds=bounds, options=options)
            _, model = fit_feasible(dace, np.atleast_1d(res.x), relocate=True)
            optima.append(model)
            if record:
                history.append(model)

        # MLE selects the best converged optimum -- unchanged behavior. Validation
        # ranks the FULL recorded search history, so it chooses among every theta the
        # gradient search visited (like Boxmin), not just the per-restart optima --
        # which means selection is meaningful even when n_restarts=0.
        return self._select(dace, history if record else optima), None
