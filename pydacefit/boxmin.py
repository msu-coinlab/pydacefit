import numpy as np

from pydacefit.fit import fit


def start(theta, dace):
    t, lo, up, p = theta, dace.tl, dace.tu, 1
    D = np.power(2, np.arange(1, p + 1).T / (p + 2))

    # if the equality constraint is equal then no search necessary
    ee = np.where(lo == up)[0]
    if len(ee) > 0:
        D[ee] = 1
        t[ee] = up[ee]

    # if theta is not in bounds - bring it in bounds
    ng = np.where(np.logical_or(t < lo, up < t))[0]
    if len(ng) > 0:
        t[ng] = np.power(lo[ng] * np.power(up[ng], 7), (1 / 8))

    ne = np.where(D != 1)[0]

    X, Y = dace.model["nX"], dace.model["nY"]
    model = fit(X, Y, dace.regr, dace.kernel, t)

    if type(lo) is not np.ndarray:
        lo = np.array([lo])

    if type(up) is not np.ndarray:
        up = np.array([up])

    itpar = {
        'best': model,
        'models': [model],
        'D': D,
        'ne': ne,
        'lo': lo,
        'up': up,
        'p': len(lo)
    }

    # try to improve starting guess if out of bounds
    if len(ng) > 0:
        raise Exception("Not implemented yet.")

    return itpar


def evaluate_and_set_best(tt, dace, itpar):
    # evaluate the model and append
    model = fit(dace.model["nX"], dace.model["nY"], dace.regr, dace.kernel, tt)
    itpar["models"].append(model)

    # flag the model as best if it is
    best = itpar["best"]
    if model["f"] < best["f"]:
        itpar["best"] = model
        return True
    else:
        return False


def explore(dace, itpar):
    t = itpar["best"]["theta"]
    ne, D, lo, up = itpar["ne"], itpar["D"], itpar["lo"], itpar["up"]

    for k in range(len(ne)):

        j = ne[k]
        tt = np.copy(t)
        DD = D[j]

        if t[j] == up[j]:
            tt[j] = t[j] / np.sqrt(DD)
        elif t[j] == lo[j]:
            tt[j] = t[j] * np.sqrt(DD)
        else:
            tt[j] = min(up[j], t[j] * DD)

        # first try to increase theta
        has_improved = evaluate_and_set_best(tt, dace, itpar)

        # if not improvement then decrease theta
        if not has_improved:

            # if the bounds were not reached
            if t[j] != up[j] and t[j] != lo[j]:
                tt = np.copy(t)
                tt[j] = max(lo[j], t[j] / DD)

                has_improved = evaluate_and_set_best(tt, dace, itpar)

        return has_improved


def move(th, dace, itpar):

    p, ne, D, lo, up = itpar["p"], itpar["ne"], itpar["D"], itpar["lo"], itpar["up"]

    t = itpar["best"]["theta"]
    v = t / th

    cont = True

    while cont:
        t = itpar["best"]["theta"]

        tt = t * v
        tt[tt > up] = up[tt > up]
        tt[tt < lo] = lo[tt < lo]

        cont = evaluate_and_set_best(tt, dace, itpar)
        if cont:
            v = np.power(v, 2)

        if np.any(np.logical_or(tt == lo, tt == up)):
            cont = False

    itpar["D"] = np.power(itpar["D"], 0.25)
