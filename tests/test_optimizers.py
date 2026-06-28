"""Tests for theta optimizers: analytic gradient correctness, L-BFGS-B, list-theta fix."""

import numpy as np

from pydacefit.corr import Correlation, Gaussian
from pydacefit.dace import DACE
from pydacefit.fit import fit
from pydacefit.optimizers import LBFGS, objective_gradient
from pydacefit.regr import ConstantRegression

# shared stateless instances (kernels/trends carry no fit state)
GAUSS = Gaussian()
CONSTANT = ConstantRegression()


def _standardized(seed, d, n=18):
    # build standardized (nX, nY) the same way DACE.fit does internally
    rng = np.random.default_rng(seed)
    X = rng.random((n, d))
    y = np.sum(np.sin(X * 3.0), axis=1)[:, None]
    nX = (X - X.mean(0)) / X.std(0, ddof=1)
    nY = (y - y.mean(0)) / y.std(0, ddof=1)
    return nX, nY


def _obj(nX, nY, theta):
    return fit(nX, nY, CONSTANT, GAUSS, theta)["f"]


def _fd_grad(nX, nY, theta, eps=1e-6):
    # central finite difference of the objective w.r.t. each theta component
    theta = np.atleast_1d(np.array(theta, dtype=float))
    g = np.zeros_like(theta)
    for k in range(len(theta)):
        tp, tm = theta.copy(), theta.copy()
        tp[k] += eps
        tm[k] -= eps
        g[k] = (_obj(nX, nY, tp) - _obj(nX, nY, tm)) / (2 * eps)
    return g


# --- the core correctness check: analytic gradient == finite differences ---


def test_gauss_isotropic_gradient_matches_finite_difference():
    # multi-dim input with a single (length-1) theta -> exercises the sum-over-dims
    # isotropic branch of Gaussian.theta_grad.
    nX, nY = _standardized(0, 3)
    theta = np.array([0.7])
    model = fit(nX, nY, CONSTANT, GAUSS, theta)

    analytic = objective_gradient(nX, model, theta, GAUSS.theta_grad)
    finite = _fd_grad(nX, nY, theta)

    assert analytic.shape == (1,)
    assert np.allclose(analytic, finite, rtol=1e-5, atol=1e-7)


def test_gauss_ard_gradient_matches_finite_difference():
    # vector theta (one per input dim) -> per-dim ARD branch.
    nX, nY = _standardized(1, 3)
    theta = np.array([0.5, 1.3, 0.9])
    model = fit(nX, nY, CONSTANT, GAUSS, theta)

    analytic = objective_gradient(nX, model, theta, GAUSS.theta_grad)
    finite = _fd_grad(nX, nY, theta)

    assert analytic.shape == (3,)
    assert np.allclose(analytic, finite, rtol=1e-5, atol=1e-7)


def test_gradient_matches_fd_across_several_thetas():
    # robustness: the match must hold away from any one lucky point.
    nX, nY = _standardized(4, 2)
    for theta in (np.array([0.2, 0.2]), np.array([1.0, 3.0]), np.array([5.0, 0.4])):
        model = fit(nX, nY, CONSTANT, GAUSS, theta)
        analytic = objective_gradient(nX, model, theta, GAUSS.theta_grad)
        finite = _fd_grad(nX, nY, theta)
        assert np.allclose(analytic, finite, rtol=1e-4, atol=1e-6), theta


# --- the analytic-gradient detection wiring ---


def test_kernel_reports_whether_it_has_an_analytic_theta_grad():
    # the kernel answers for itself now (no free helper): Gaussian provides one...
    assert GAUSS.has_theta_grad is True

    # ...a kernel that implements neither hook inherits the raising base, so it reports
    # False and LBFGS falls back to scipy's numeric gradient instead.
    class MadeUp(Correlation):
        def __call__(self, D, theta):
            return np.ones(D.shape[0])

    assert MadeUp().has_theta_grad is False


# --- end-to-end L-BFGS-B with the analytic gradient ---


def test_lbfgs_improves_objective_and_converges_interior():
    rng = np.random.default_rng(2)
    X = rng.random((25, 2))
    y = np.sum(np.sin(X * 3.0), axis=1)

    lo, up = 1e-4, 50.0
    # tight tolerance here so the gradient-at-optimum assertion is meaningful
    model = DACE(
        regr=CONSTANT,
        corr=GAUSS,
        theta=np.array([1.0, 1.0]),
        thetaL=[lo, lo],
        thetaU=[up, up],
        optimizer=LBFGS(options={"gtol": 1e-8, "ftol": 1e-12}),
    )
    model.fit(X, y)

    # the optimum must not be worse than the start point
    start_obj = fit(model.model["nX"], model.model["nY"], CONSTANT, GAUSS, np.array([1.0, 1.0]))["f"]
    assert model.model["f"] <= start_obj + 1e-9

    # for theta components strictly inside the bounds, the gradient must be ~0
    theta = np.ravel(model.model["theta"])
    grad = objective_gradient(model.model["nX"], model.model, theta, GAUSS.theta_grad)
    interior = (theta > lo * 1.01) & (theta < up * 0.99)
    assert np.all(np.abs(grad[interior]) < 1e-3)
    assert np.all(np.isfinite(model.predict(rng.random((5, 2))).y))


def test_relaxed_lbfgs_saves_evaluations_and_stays_close():
    # the whole point of the relaxed default: fewer objective evaluations than a
    # tight run, while still landing on essentially the same optimum.
    import pydacefit.optimizers.lbfgs as lb

    rng = np.random.default_rng(3)
    X = rng.random((22, 2))
    y = np.sum(np.sin(X * 3.0), axis=1)

    def _build(optimizer):
        return DACE(
            regr=CONSTANT,
            corr=GAUSS,
            theta=np.array([1.0, 1.0]),
            thetaL=[1e-4, 1e-4],
            thetaU=[50.0, 50.0],
            optimizer=optimizer,
        )

    calls = [0]
    real_fit = lb.fit

    def counting(*a, **k):
        calls[0] += 1
        return real_fit(*a, **k)

    lb.fit = counting
    try:
        calls[0] = 0
        relaxed = _build(LBFGS(options={"gtol": 1e-2}))
        relaxed.fit(X, y)
        relaxed_evals = calls[0]

        calls[0] = 0
        tight = _build(LBFGS(options={"gtol": 1e-9, "ftol": 1e-14}))
        tight.fit(X, y)
        tight_evals = calls[0]
    finally:
        lb.fit = real_fit

    assert relaxed_evals < tight_evals  # relaxing the tolerance really saves work
    assert relaxed.model["f"] <= tight.model["f"] * 1.02 + 1e-9  # still a good optimum


def test_lbfgs_matches_boxmin_when_optimum_is_unique():
    # on a well-conditioned 1d problem both optimizers find the same single optimum.
    rng = np.random.default_rng(11)
    X = rng.random((20, 1))
    y = np.sum(np.sin(X * 2 * np.pi), axis=1)

    box = DACE(regr=CONSTANT, corr=GAUSS, theta=1.0, thetaL=1e-4, thetaU=100.0)
    box.fit(X, y)
    lb = DACE(regr=CONSTANT, corr=GAUSS, theta=1.0, thetaL=1e-4, thetaU=100.0, optimizer=LBFGS())
    lb.fit(X, y)

    xt = np.linspace(0, 1, 40)[:, None]
    assert np.allclose(box.predict(xt).y, lb.predict(xt).y, atol=1e-2)


def test_lbfgs_restarts_escape_a_bad_starting_basin():
    # the DACE likelihood is multi-modal: from a poor start theta (here the lower
    # bound, a flat plateau where the gradient vanishes) a single L-BFGS-B run stays
    # stuck. Random restarts must escape it and reach the good optimum.
    rng = np.random.default_rng(3)
    X = rng.random((30, 1))
    y = np.sin(X[:, 0] * 12.0)

    bad_start = 1e-4  # the lower bound -> single start cannot move off the plateau
    single = DACE(regr=CONSTANT, corr=GAUSS, theta=bad_start, thetaL=1e-4, thetaU=100.0, optimizer=LBFGS())
    single.fit(X, y)

    restarts = DACE(
        regr=CONSTANT, corr=GAUSS, theta=bad_start, thetaL=1e-4, thetaU=100.0, optimizer=LBFGS(n_restarts=12)
    )
    restarts.fit(X, y)

    # restarts find a far better objective and recover the (near-interpolating) fit
    assert restarts.model["f"] < single.model["f"] - 1.0
    assert restarts.model["f"] < 1e-3
    assert restarts.model["theta"].ravel()[0] > bad_start  # actually moved off the plateau


def test_lbfgs_first_start_is_the_configured_theta():
    # n_restarts must not change the warm-start contract: the first start is always
    # the configured theta, so a feasible warm start can only be improved upon.
    rng = np.random.default_rng(5)
    X = rng.random((20, 1))
    y = np.sum(np.sin(X * 2 * np.pi), axis=1)

    plain = DACE(regr=CONSTANT, corr=GAUSS, theta=1.0, thetaL=1e-4, thetaU=100.0, optimizer=LBFGS())
    plain.fit(X, y)
    multi = DACE(regr=CONSTANT, corr=GAUSS, theta=1.0, thetaL=1e-4, thetaU=100.0, optimizer=LBFGS(n_restarts=5))
    multi.fit(X, y)

    # multi-start keeps the best, so it is never worse than the single warm start
    assert multi.model["f"] <= plain.model["f"] + 1e-9


def test_lbfgs_default_options_relaxed_with_bounded_evals():
    # the bare optimizer carries relaxed tolerances and a low evaluation cap, since
    # each objective evaluation runs a full O(n^3) fit.
    assert LBFGS().options == {"gtol": 1e-3, "ftol": 1e-6, "maxfun": 100}


def test_lbfgs_options_passthrough_reaches_scipy():
    # scipy L-BFGS-B options pass through verbatim, and a key given in `options`
    # overrides the relaxed default (here gtol) while untouched defaults remain.
    import pydacefit.optimizers.lbfgs as lb

    rng = np.random.default_rng(7)
    X = rng.random((18, 1))
    y = np.sum(np.sin(X * 3.0), axis=1)

    seen = []
    real_minimize = lb.minimize

    def recording(*a, **k):
        seen.append(k.get("options"))
        return real_minimize(*a, **k)

    lb.minimize = recording
    try:
        model = DACE(
            regr=CONSTANT,
            corr=GAUSS,
            theta=1.0,
            thetaL=1e-4,
            thetaU=100.0,
            optimizer=LBFGS(options={"maxiter": 7, "gtol": 1e-1}),
        )
        model.fit(X, y)
    finally:
        lb.minimize = real_minimize

    assert seen  # minimize was actually called
    opts = seen[0]
    assert opts["maxiter"] == 7  # extra option forwarded through to scipy
    assert opts["gtol"] == 1e-1  # repeated key overrides the relaxed default
    assert opts["ftol"] == 1e-6 and opts["maxfun"] == 100  # untouched defaults remain
    assert np.all(np.isfinite(model.predict(rng.random((4, 1))).y))


def test_lbfgs_exposes_optimization_record():
    # the fit must surface the optimizer result on model.optimization: eval counts,
    # per-start scipy results, and a start count that tracks n_restarts.
    rng = np.random.default_rng(9)
    X = rng.random((20, 1))
    y = np.sum(np.sin(X * 3.0), axis=1)

    model = DACE(regr=CONSTANT, corr=GAUSS, theta=1.0, thetaL=1e-4, thetaU=100.0, optimizer=LBFGS(n_restarts=3))
    model.fit(X, y)

    rec = model.optimization
    assert rec is not None
    assert rec["n_starts"] == 4  # the warm start + 3 restarts
    assert len(rec["results"]) == 4  # one scipy OptimizeResult per start
    assert rec["nit"] >= 1 and rec["nfev"] >= 1  # aggregate iteration / eval counts
    assert isinstance(rec["success"], bool)
    assert all(hasattr(r, "nit") and hasattr(r, "message") for r in rec["results"])  # raw scipy results
    assert rec["best"] is model.model or "gamma" in rec["best"]  # the chosen fit is in the record


# --- pre-existing bug fix: a Python-list theta (ARD) no longer crashes boxmin ---


def test_list_theta_ard_fits_via_boxmin():
    # regression: a list theta used to reach the kernel as a list and crash with
    # "bad operand type for unary -: 'list'"; __init__ now coerces it to an array.
    rng = np.random.default_rng(0)
    X = rng.random((20, 2))
    y = np.sum(np.sin(X * 3.0), axis=1)

    model = DACE(regr=CONSTANT, corr=GAUSS, theta=[1.0, 1.0], thetaL=[1e-4, 1e-4], thetaU=[20.0, 20.0])
    model.fit(X, y)
    assert np.all(np.isfinite(model.predict(rng.random((4, 2))).y))


def test_lbfgs_restarts_handle_zero_lower_bound():
    # regression: LBFGS(n_restarts>0) sampled starts as 10**uniform(log10(lo), ...),
    # so DACE's default thetaL=0.0 gave log10(0)=-inf -> NaN starts. The lower bound
    # is now floored away from zero, so restarts stay finite and the fit succeeds.
    rng = np.random.default_rng(0)
    X = rng.random((20, 1))
    y = np.sum(np.sin(X * 3.0), axis=1)

    model = DACE(regr=CONSTANT, corr=GAUSS, theta=1.0, optimizer=LBFGS(n_restarts=4))
    model.fit(X, y)

    assert np.all(np.isfinite(np.ravel(model.model["theta"])))
    assert np.all(np.isfinite(model.predict(rng.random((5, 1))).y))
