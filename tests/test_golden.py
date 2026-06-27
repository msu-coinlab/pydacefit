"""Golden behavior-oracle: tight snapshots of DACE across its well-conditioned envelope.

Spans all seven kernels, the three regression trends, the optimization and the
fixed-theta paths, scalar and vector (ARD) theta, and the prediction, MSE and
gradient outputs. Inputs are kept well-conditioned on purpose — degenerate /
ill-conditioned behavior is pinned as a contract in test_degenerate.py, because
those outputs are numerically unstable and unfit for a tight cross-platform baseline.
"""

import numpy as np
import pytest

from pydacefit.corr import (
    corr_cubic,
    corr_exp,
    corr_expg,
    corr_gauss,
    corr_lin,
    corr_rq,
    corr_spherical,
    corr_spline,
)
from pydacefit.dace import DACE
from pydacefit.regr import regr_constant, regr_linear, regr_quadratic


def _data():
    # deterministic, well-spread 2-D grid (4x3 = 12 points) — no RNG, and spaced
    # enough to keep even the compact-support kernels well-conditioned.
    gx, gy = np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, 1, 3))
    x_train = np.column_stack([gx.ravel(), gy.ravel()])
    f_train = np.sin(x_train[:, 0] * 3) + np.cos(x_train[:, 1] * 2)
    x_test = np.column_stack([np.linspace(0.1, 0.9, 5), np.linspace(0.2, 0.8, 5)])
    return x_train, f_train, x_test


# scalar-theta optimization bounds and ARD (vector) bounds reused across cases
_OPT = dict(theta=0.5, thetaL=0.05, thetaU=10.0)
_NOOPT = dict(theta=0.5, thetaL=None, thetaU=None)
_ARD = dict(theta=np.array([0.5, 0.5]), thetaL=np.array([0.05, 0.05]), thetaU=np.array([10.0, 10.0]))
# expg's theta is (length-scale, power); both forms below keep it well-conditioned
_EXPG = dict(theta=np.array([0.5, 2.0]), thetaL=None, thetaU=None)
_EXPG_OPT = dict(theta=np.array([0.5, 2.0]), thetaL=np.array([0.05, 1.0]), thetaU=np.array([10.0, 3.0]))

# (id, regr, corr, params) — chosen to cover every kernel x regression x path x theta-form
CASES = [
    ("gauss/const/opt-scalar", regr_constant, corr_gauss, _OPT),
    ("gauss/linear/noopt", regr_linear, corr_gauss, _NOOPT),
    ("gauss/quad/opt-scalar", regr_quadratic, corr_gauss, _OPT),
    ("gauss/const/opt-ard", regr_constant, corr_gauss, _ARD),
    ("exp/const/opt-scalar", regr_constant, corr_exp, _OPT),
    ("exp/linear/noopt", regr_linear, corr_exp, _NOOPT),
    ("cubic/const/opt-scalar", regr_constant, corr_cubic, _OPT),
    ("cubic/quad/noopt", regr_quadratic, corr_cubic, _NOOPT),
    ("spline/const/noopt", regr_constant, corr_spline, _NOOPT),
    ("spherical/const/opt-ard", regr_constant, corr_spherical, _ARD),
    ("spherical/linear/opt-scalar", regr_linear, corr_spherical, _OPT),
    ("lin/quad/noopt", regr_quadratic, corr_lin, _NOOPT),
    ("expg/const/noopt", regr_constant, corr_expg, _EXPG),
    ("expg/const/opt-ard", regr_constant, corr_expg, _EXPG_OPT),
    ("rq/const/opt-scalar", regr_constant, corr_rq, _OPT),
    ("rq/quad/opt-ard", regr_quadratic, corr_rq, _ARD),
]


@pytest.mark.golden
@pytest.mark.parametrize("name,regr,corr,params", CASES, ids=[c[0] for c in CASES])
def test_predict(name, regr, corr, params):
    x_train, f_train, x_test = _data()
    model = DACE(regr=regr, corr=corr, **params)
    model.fit(x_train, f_train)

    pred, mse, grad = model.predict(x_test, return_mse=True, return_gradient=True)

    snapshot = {
        "pred": pred.ravel(),
        "mse": mse.ravel(),
        "grad": grad.ravel(),
        "theta": np.asarray(model.model["theta"]).ravel(),
    }
    # for the optimization path also pin the full theta search trajectory
    if model.itpar is not None:
        snapshot["theta_traj"] = np.stack([m["theta"] for m in model.itpar["models"]]).ravel()
    return snapshot
