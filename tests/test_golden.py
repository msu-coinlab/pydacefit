"""Golden behavior-oracle: tight snapshots of DACE across its well-conditioned envelope.

Spans all nine kernels (Matérn at each smoothness nu), the three regression trends,
the optimization and the fixed-theta paths, scalar and vector (ARD) theta, and the
prediction, MSE and gradient outputs. Inputs are kept well-conditioned on purpose —
degenerate / ill-conditioned behavior is pinned as a contract in test_degenerate.py,
because those outputs are numerically unstable and unfit for a tight cross-platform
baseline.
"""

import numpy as np
import pytest

from pydacefit.corr import (
    Cubic,
    Exponential,
    Gaussian,
    GeneralizedExponential,
    Linear,
    Matern,
    RationalQuadratic,
    Spherical,
    Spline,
)
from pydacefit.dace import DACE
from pydacefit.regr import ConstantRegression, LinearRegression, QuadraticRegression


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
    ("gauss/const/opt-scalar", ConstantRegression(), Gaussian(), _OPT),
    ("gauss/linear/noopt", LinearRegression(), Gaussian(), _NOOPT),
    ("gauss/quad/opt-scalar", QuadraticRegression(), Gaussian(), _OPT),
    ("gauss/const/opt-ard", ConstantRegression(), Gaussian(), _ARD),
    ("exp/const/opt-scalar", ConstantRegression(), Exponential(), _OPT),
    ("exp/linear/noopt", LinearRegression(), Exponential(), _NOOPT),
    ("cubic/const/opt-scalar", ConstantRegression(), Cubic(), _OPT),
    ("cubic/quad/noopt", QuadraticRegression(), Cubic(), _NOOPT),
    ("spline/const/noopt", ConstantRegression(), Spline(), _NOOPT),
    ("spherical/const/opt-ard", ConstantRegression(), Spherical(), _ARD),
    ("spherical/linear/opt-scalar", LinearRegression(), Spherical(), _OPT),
    ("lin/quad/noopt", QuadraticRegression(), Linear(), _NOOPT),
    ("expg/const/noopt", ConstantRegression(), GeneralizedExponential(), _EXPG),
    ("expg/const/opt-ard", ConstantRegression(), GeneralizedExponential(), _EXPG_OPT),
    # pin alpha explicitly so the baseline is independent of the RQ default (now 0.25)
    ("rq/const/opt-scalar", ConstantRegression(), RationalQuadratic(alpha=1.0), _OPT),
    ("rq/quad/opt-ard", QuadraticRegression(), RationalQuadratic(alpha=1.0), _ARD),
    # Matérn at each closed-form smoothness; pin nu so the baseline is default-independent
    ("matern25/const/opt-scalar", ConstantRegression(), Matern(nu=2.5), _OPT),
    ("matern15/linear/noopt", LinearRegression(), Matern(nu=1.5), _NOOPT),
    ("matern05/const/noopt", ConstantRegression(), Matern(nu=0.5), _NOOPT),
]


@pytest.mark.golden
@pytest.mark.parametrize("name,regr,corr,params", CASES, ids=[c[0] for c in CASES])
def test_predict(name, regr, corr, params):
    x_train, f_train, x_test = _data()
    model = DACE(regr=regr, corr=corr, **params)
    model.fit(x_train, f_train)

    p = model.predict(x_test, mse=True, grad=True)

    snapshot = {
        "pred": p.y.ravel(),
        "mse": p.mse.ravel(),
        "grad": p.grad.ravel(),
        "theta": np.asarray(model.model["theta"]).ravel(),
    }
    # for the optimization path also pin the full theta search trajectory
    if model.optimization is not None:
        snapshot["theta_traj"] = np.stack([m["theta"] for m in model.optimization["models"]]).ravel()
    return snapshot
