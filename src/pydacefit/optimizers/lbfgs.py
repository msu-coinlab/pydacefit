"""L-BFGS-B optimizer with an analytic theta-gradient (when the kernel provides one)."""

import numpy as np
from scipy.optimize import minimize  # type: ignore[import-untyped]

from pydacefit.fit import DaceFitError, fit
from pydacefit.optimizers.base import Optimizer, fit_feasible

# returned by a failed fit so the gradient-based optimizer sees a large but finite
# penalty (np.inf breaks finite-difference gradients) and steps away from it.
_INFEASIBLE = 1e25


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
    (``kernel.has_theta_grad``; all shipped kernels do) it is used as the exact
    Jacobian -- this is what makes it fast; otherwise it falls back to scipy's
    finite-difference gradient.

    The default stop tolerances are deliberately *relaxed* (``gtol=1e-3``,
    ``ftol=1e-6``): a warm-started refit already sits next to the optimum, so chasing
    the last digits costs iterations without meaningfully changing the model. Loosen
    further (``LBFGS(options={"gtol": 1e-2})``) to save more, or tighten
    (``LBFGS(options={"gtol": 1e-8})``) when an exact optimum matters.

    The DACE likelihood is multi-modal, so a single local search can settle in a
    poor basin when started far from the optimum. ``n_restarts`` adds that many
    random log-uniform restarts within the bounds and keeps the best result. The
    first start is always the model's own (warm) theta, so ``n_restarts=0`` keeps
    the warm-start behavior unchanged -- restarts are only useful for cold fits from
    an unknown scale, where the configured start may miss the global optimum.

    Parameters
    ----------
    n_restarts : int
        Number of additional random restarts (beyond the configured start) to guard
        against local optima. 0 (default) means a single start from the model theta.

    random_state : int or None
        Seed for the restart sampling, so multi-start fits are reproducible.

    options : dict or None
        Options forwarded to ``scipy.optimize.minimize(..., method="L-BFGS-B",
        options=...)``. Any key the solver accepts works -- ``gtol``, ``ftol``,
        ``maxiter``, ``maxfun``, ``eps``, ``maxcor``, ``maxls``. The defaults
        ``{"gtol": 1e-3, "ftol": 1e-6, "maxfun": 100}`` apply relaxed stop tolerances
        and cap the objective evaluations (the expensive O(n^3) fits) at a level that
        only stops a runaway search; anything passed here overrides or extends them.
        None (default) uses the defaults alone.

    The held-out set is passed to ``optimize`` (by ``DACE.fit``), not to the
    constructor. When given, theta is chosen by ranking the *entire* search history --
    every theta the gradient search evaluates is recorded (at no extra cost, since the
    objective already fits a model there) and the one with the lowest held-out error is
    kept. So selection is meaningful even with ``n_restarts=0``, where a single descent
    still visits many thetas. The MLE path is unaffected: with no validation set the
    best per-restart optimum is returned, exactly as before.
    """

    def __init__(self, n_restarts=0, random_state=0, options=None):
        super().__init__()
        self.n_restarts = n_restarts
        self.random_state = random_state
        # relaxed stop tolerances by default -- a warm-started refit already sits near
        # the optimum. maxfun caps the number of objective evaluations, i.e. the actual
        # expensive fit() calls (each an O(n^3) Cholesky): scipy's default is 15000, but
        # LBFGS here converges in ~2-8 evals, so 100 is a large safety margin that only
        # stops a runaway/non-converging search. Override or extend with any scipy
        # L-BFGS-B option (gtol, ftol, maxiter, maxfun, eps, maxcor, maxls).
        self.options = {"gtol": 1e-3, "ftol": 1e-6, "maxfun": 100, **(options or {})}

    def optimize(self, dace, validation=None):
        """Run L-BFGS-B from the (warm) start plus any restarts; return ``(best_model, optimization)``."""
        nX, nY = dace.model["nX"], dace.model["nY"]
        regr, kernel = dace.regr, dace.kernel
        grad_func = kernel.theta_grad if kernel.has_theta_grad else None
        options = self.options

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
        record = validation is not None
        history = []

        # objective (+ analytic gradient when the kernel provides one)
        if grad_func is not None:

            def fun(t):
                t = np.array(t, dtype=float)
                try:
                    model = fit(nX, nY, regr, kernel, t, noise=dace.noise)
                except DaceFitError:
                    return _INFEASIBLE, np.zeros_like(t)
                if record:
                    history.append(model)
                return float(model["f"]), objective_gradient(nX, model, t, grad_func)

            jac = True
        else:

            def fun(t):
                try:
                    model = fit(nX, nY, regr, kernel, np.array(t, dtype=float), noise=dace.noise)
                except DaceFitError:
                    return _INFEASIBLE
                if record:
                    history.append(model)
                return float(model["f"])

            jac = False

        # the first start is always the (warm) configured theta; the rest are random
        # log-uniform within the bounds. Floor the lower bound away from zero first:
        # DACE's default thetaL is 0.0, and log10(0) = -inf would make every restart NaN.
        starts = [theta0]
        if self.n_restarts > 0:
            rng = np.random.default_rng(self.random_state)
            lo_pos = np.maximum(lo, 1e-12)
            for _ in range(self.n_restarts):
                starts.append(10.0 ** rng.uniform(np.log10(lo_pos), np.log10(up)))

        # build a model at each restart's converged theta (fit_feasible also applies
        # the shared feasibility / max_noise safety net). For n_restarts=0 this is a
        # single start, as before.
        optima = []
        results = []
        for s in starts:
            res = minimize(fun, s, method="L-BFGS-B", jac=jac, bounds=bounds, options=options)
            _, model = fit_feasible(dace, np.atleast_1d(res.x), relocate=True)
            optima.append(model)
            results.append(res)
            if record:
                history.append(model)

        # MLE selects the best converged optimum -- unchanged behavior. Validation
        # ranks the FULL recorded search history, so it chooses among every theta the
        # gradient search visited (like Boxmin), not just the per-restart optima --
        # which means selection is meaningful even when n_restarts=0.
        models = history if record else optima
        best = self._select(dace, models, validation)

        # the optimization record (lands on DACE.optimization). "best"/"models" mirror
        # Boxmin so trajectory consumers are uniform; "results" holds scipy's per-start
        # OptimizeResult (success/message/x), and nit/nfev are the totals across starts.
        optimization = {
            "best": best,
            "models": models,
            "results": results,
            "n_starts": len(starts),
            "nit": int(sum(r.nit for r in results)),
            "nfev": int(sum(r.nfev for r in results)),
            "success": all(bool(r.success) for r in results),
        }
        return best, optimization
