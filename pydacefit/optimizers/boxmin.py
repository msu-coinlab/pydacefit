"""Box-min optimizer: Hooke & Jeeves pattern search over theta (the original DACE method)."""

import numpy as np

from pydacefit.fit import fit
from pydacefit.optimizers.base import Optimizer, fit_feasible


class Boxmin(Optimizer):
    """Hooke & Jeeves pattern search -- the original DACE theta optimizer.

    Derivative-free and robust; it explores broadly from the start theta, which makes
    it the right default for a cold initial fit. ``optimize`` returns the best model
    and the full search trajectory (``itpar``).
    """

    def optimize(self, dace):
        """Run the pattern search; return ``(best_model, itpar)``."""
        itpar = _start(dace)
        model = itpar["models"][-1]
        p, f = itpar["p"], model["f"]

        kmax = 2 if p <= 2 else min(p, 4)

        # if the initial guess is feasible, run the pattern moves
        if not np.isinf(f):
            for _ in range(kmax):
                last_t = itpar["best"]["theta"]
                _explore(dace, itpar)
                _move(last_t, dace, itpar)

        return itpar["best"], itpar


def _start(dace):
    # copy so the in-place fixes below (t[ee], t[ng]) don't mutate the caller's theta
    theta = dace.theta
    t = np.copy(theta) if isinstance(theta, np.ndarray) else theta
    lo, up = dace.tl, dace.tu

    # check whether theta is a vector or not
    if type(dace.tu) is not np.ndarray:
        p = 1
    else:
        p = len(dace.tu)

    D = np.power(2, np.arange(1, p + 1).T / (p + 2))

    # if the equality constraint is equal then no search necessary
    # (atleast_1d so scalar theta bounds still produce a 1d index array)
    ee = np.atleast_1d(lo == up).nonzero()[0]
    if len(ee) > 0:
        D[ee] = 1
        t[ee] = up[ee]

    # if theta is not in bounds - bring it in bounds
    ng = np.atleast_1d(np.logical_or(t < lo, up < t)).nonzero()[0]
    if len(ng) > 0:
        t[ng] = np.power(lo[ng] * np.power(up[ng], 7), (1 / 8))

    ne = np.where(D != 1)[0]

    # fit at t; if it sits in a non-positive-definite pocket, relocate to a feasible
    # theta (toward thetaU) instead of crashing the Cholesky inside fit()
    t, model = fit_feasible(dace, t, relocate=True)

    if type(lo) is not np.ndarray:
        lo = np.array([lo])

    if type(up) is not np.ndarray:
        up = np.array([up])

    itpar = {"best": model, "models": [model], "D": D, "ne": ne, "lo": lo, "up": up, "p": len(lo)}

    # try to improve starting guess if out of bounds
    if len(ng) > 0:
        raise Exception("Theta should always be in bounds in this implementation. Not implemented yet.")

    return itpar


def _evaluate_and_set_best(tt, dace, itpar):
    # evaluate the model and append; a non-PD theta is infeasible -> infinite objective
    try:
        model = fit(dace.model["nX"], dace.model["nY"], dace.regr, dace.kernel, tt)
        itpar["models"].append(model)

    except Exception:
        itpar["models"].append({"theta": tt, "obj": np.inf})
        return False

    # flag the model as best if it is
    best = itpar["best"]
    if model["f"] < best["f"]:
        itpar["best"] = model
        return True
    else:
        return False


def _explore(dace, itpar):
    ne, D, lo, up = itpar["ne"], itpar["D"], itpar["lo"], itpar["up"]

    for k in range(len(ne)):
        j = ne[k]

        # theta of the current best model
        t = itpar["best"]["theta"]

        DD = D[j]

        # copy the theta before modifying it
        tt = np.copy(t)

        if t[j] == up[j]:
            tt[j] = t[j] / np.sqrt(DD)
        elif t[j] == lo[j]:
            tt[j] = t[j] * np.sqrt(DD)
        else:
            tt[j] = min(up[j], t[j] * DD)

        # first try to increase theta
        has_improved = _evaluate_and_set_best(tt, dace, itpar)

        # if not improvement then decrease theta
        if not has_improved:
            # if the bounds were not reached
            if t[j] != up[j] and t[j] != lo[j]:
                tt = np.copy(t)
                tt[j] = max(lo[j], t[j] / DD)

                _evaluate_and_set_best(tt, dace, itpar)


def _move(th, dace, itpar):
    p, lo, up = itpar["p"], itpar["lo"], itpar["up"]

    # index used later on for permutation
    perm = np.concatenate([np.arange(1, p), [0]])

    t = itpar["best"]["theta"]
    v = t / th

    # if t and th is all equal
    if np.all(v == 1):
        itpar["D"] = np.power(itpar["D"][perm], 0.2)
        return

    cont = True

    while cont:
        t = itpar["best"]["theta"]

        tt = t * v
        tt[tt > up] = up[tt > up]
        tt[tt < lo] = lo[tt < lo]

        has_improved = _evaluate_and_set_best(tt, dace, itpar)

        # if it has improved we increment the step size
        if has_improved:
            v = np.power(v, 2)

        # only if improved we continue
        cont = has_improved

        # however, in case we already on the bounds we stop
        if np.any(np.logical_or(tt == lo, tt == up)):
            cont = False

    itpar["D"] = np.power(itpar["D"][perm], 0.25)
