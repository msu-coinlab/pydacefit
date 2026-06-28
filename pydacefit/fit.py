"""Generalized least-squares fit of the DACE Kriging model for a fixed theta."""

import warnings

import numpy as np
from numpy.linalg import LinAlgError

from pydacefit.corr import calc_kernel_matrix


class DaceFitError(Exception):
    """A fit is infeasible at this theta (non-positive-definite R, or ill-conditioned F).

    Raised instead of a bare ``Exception`` so theta optimizers can ``except`` it and
    treat the offending theta as an infinite objective -- rejecting the step instead
    of letting the failure abort the whole search. An infeasible theta should *score
    badly*, not crash.
    """


def fit(X, Y, regr, kernel, theta, noise=0.0, max_noise=0.0):

    # number of sample points (rows of the design matrix)
    n_sample = X.shape[0]

    # baseline float-level jitter against near-singularity, always added -- the invisible
    # numerical floor (machine epsilon scaled by sample size), not a modeling choice
    base = (10 + n_sample) * 2.220446049250313e-16
    R0 = calc_kernel_matrix(X, X, kernel, theta)

    # do the cholesky decomposition. The diagonal carries the DELIBERATE observation
    # `noise` (a noise-to-signal ratio on R's unit diagonal: 0 -> interpolate, >0 ->
    # regression GP that smooths through points), always added. If R is still not
    # positive-definite, a SEPARATE `repair` term climbs from a tiny floor up to
    # `max_noise` -- independent of `noise`, so the repair budget works regardless of how
    # much deliberate noise was set -- until R factorizes. max_noise=0 permits no repair,
    # so a non-PD R raises and the optimizers read that as an infeasible theta. If even
    # max_noise cannot make R PD, we stop and raise. The recorded model["noise"] is the
    # total effective diagonal (deliberate + repair).
    repair = 0.0
    while True:
        R = R0 + np.eye(n_sample) * (base + noise + repair)
        try:
            C = np.linalg.cholesky(R)
            break
        except LinAlgError as e:
            nxt = repair * 2.0 if repair > 0.0 else 1e-12
            if nxt > max_noise:
                raise DaceFitError("Error while doing Cholesky Decomposition.") from e
            repair = nxt

    noise = noise + repair
    if repair > 0.0:
        warnings.warn(
            f"R was not positive-definite at the requested noise; added repair={repair:g} "
            f"(<= max_noise) for a total noise of {noise:g} -- a regularized fit, not an "
            f"exact interpolator.",
            stacklevel=2,
        )

    # fit the least squares for regression
    F = regr(X)
    Ft = np.linalg.lstsq(C, F, rcond=None)[0]
    Q, G = np.linalg.qr(Ft)
    rcond = 1.0 / np.linalg.cond(G)
    if rcond > 1e15:
        raise DaceFitError("F is too ill conditioned: Poor combination of regression model and design sites")
    Yt = np.linalg.solve(C, Y)
    beta = np.linalg.lstsq(G, Q.T @ Yt, rcond=None)[0]

    # calculate the residual to fit with gaussian process and calculate objective function
    rho = Yt - Ft @ beta
    sigma2 = np.sum(np.square(rho), axis=0) / n_sample
    detR = np.prod(np.power(np.diag(C), (2 / n_sample)))
    obj = np.sum(sigma2) * detR

    # finally gamma to predict values
    gamma = np.linalg.solve(C.T, rho)

    if type(theta) is not np.ndarray:
        theta = np.array([theta])

    return {
        "kernel": kernel,
        "regr": regr,
        "theta": theta,
        "R": R,
        "C": C,
        "F": F,
        "Ft": Ft,
        "Q": Q,
        "G": G,
        "Yt": Yt,
        "beta": beta,
        "rho": rho,
        "_sigma2": sigma2,
        "obj": obj,
        "f": obj,
        "gamma": gamma,
        "noise": noise,
    }
