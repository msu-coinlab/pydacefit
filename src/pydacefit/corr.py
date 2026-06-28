"""Correlation (kernel) functions for the DACE model, as first-class objects."""

import numpy as np


# function to calculate the correlation matrix all in one
def calc_kernel_matrix(A, B, func, theta):
    D = np.repeat(A, B.shape[0], axis=0) - np.tile(B, (A.shape[0], 1))
    K = func(D, theta)
    return np.reshape(K, (A.shape[0], B.shape[0]))


def calc_grad(A, B, func, theta):
    D = np.repeat(A, B.shape[0], axis=0) - np.tile(B, (A.shape[0], 1))
    return func(D, theta)


# -------------------------------
# Correlation Kernels
# -------------------------------


class Correlation:
    """A correlation kernel: a callable object bundling its own gradients.

    A kernel maps componentwise distances ``D`` and length-scales ``theta`` to a
    correlation vector, and carries the two derivatives the model may need:

    - ``__call__(D, theta)`` -- the correlation itself.
    - ``grad(D, theta)`` -- derivative w.r.t. the design point, used by
      ``predict(grad=True)``.
    - ``theta_grad(D, theta)`` -- derivative w.r.t. ``theta``, one column per scalar in
      ``theta``, used by gradient-based optimizers (e.g. ``LBFGS``).

    ``theta_grad`` is implemented here once and delegates to the optional
    ``_dtheta_per_dim`` hook (the per-dimension theta partials); this base collapses the
    isotropic case so kernels don't each repeat it. A kernel supplies just that hook --
    or overrides ``theta_grad`` outright when its ``theta`` is laid out differently (e.g.
    ``GeneralizedExponential``, which also tunes an exponent). ``grad`` / ``theta_grad``
    are *optional*: a kernel implementing neither leaves ``theta_grad`` raising
    ``NotImplementedError`` and gradient-based consumers fall back (numerical gradient /
    derivative-free ``Boxmin``). ``theta`` carries the length-scale(s) -- scalar
    (isotropic) or per-dimension vector (ARD) -- and is the only model-tuned parameter;
    any fixed shape parameter (e.g. RQ's ``alpha``) lives on the kernel object instead.
    """

    def __call__(self, D, theta):
        raise NotImplementedError

    def grad(self, D, theta):
        raise NotImplementedError

    def theta_grad(self, D, theta):
        # one column per scalar in theta. ARD (vector theta) keeps the per-dimension
        # partials; isotropic (a single shared theta) is, by the chain rule, the sum of
        # those partials -> a single column. The kernel only supplies _dtheta_per_dim.
        per_dim = self._dtheta_per_dim(D, theta)
        if np.size(theta) == 1:
            return per_dim.sum(axis=1, keepdims=True)
        return per_dim

    def _dtheta_per_dim(self, D, theta):
        # optional analytic hook: d corr / d theta_k for each input dimension k,
        # shape (n_pairs, d). Left unimplemented -> the kernel has no analytic theta
        # gradient and LBFGS uses a numeric one instead.
        raise NotImplementedError

    @property
    def has_theta_grad(self):
        """Whether this kernel provides an analytic theta-gradient.

        True if it implements the ``_dtheta_per_dim`` hook (the usual path) or overrides
        ``theta_grad`` outright (e.g. ``GeneralizedExponential``). A gradient-based
        optimizer asks the kernel this to choose between the exact Jacobian and a
        finite-difference fallback.
        """
        cls = type(self)
        return cls.theta_grad is not Correlation.theta_grad or cls._dtheta_per_dim is not Correlation._dtheta_per_dim

    def __repr__(self):
        return type(self).__name__


class Gaussian(Correlation):
    """Gaussian (squared-exponential) kernel: ``exp(-sum_k theta_k * D_k**2)``."""

    def __call__(self, D, theta):
        return np.exp(np.sum(np.square(D) * -theta, axis=1))

    def grad(self, D, theta):
        return -2 * theta * D * self(D, theta)[:, None]

    def _dtheta_per_dim(self, D, theta):
        # exp(-sum theta_k d_k^2) factorizes, so d/d(theta_k) = -d_k^2 * corr.
        return -np.square(D) * self(D, theta)[:, None]


class Cubic(Correlation):
    """Cubic kernel: compact-support smooth correlation (zero past the length scale)."""

    def __call__(self, D, theta):
        td = np.minimum(np.abs(D) * theta, 1)
        ss = 1 - td**2 * (3 - 2 * td)
        r = np.prod(ss, axis=1)
        return r

    def grad(self, D, theta):
        dr = np.zeros(D.shape)
        td = np.minimum(np.abs(D) * theta, 1)
        ss = 1 - td**2 * (3 - 2 * td)

        for j in range(D.shape[1]):
            _b = index_except(D.shape[1], [j])
            _theta = theta[j] if type(theta) is np.ndarray and len(theta) == D.shape[1] else theta
            dd = 6 * _theta * np.sign(D[:, j]) * td[:, j] * (td[:, j] - 1)

            dr[:, j] = np.prod(ss[:, _b], axis=1) * dd
        return dr

    def _dtheta_per_dim(self, D, theta):
        td = np.minimum(np.abs(D) * theta, 1)
        ss = 1 - td**2 * (3 - 2 * td)
        dr = np.zeros(D.shape)
        for j in range(D.shape[1]):
            _b = index_except(D.shape[1], [j])
            # d s_j/d theta_j = (d s/d t)(d t/d theta) = 6 t(t-1) * |D_j| (0 at the clamp)
            dd = 6 * td[:, j] * (td[:, j] - 1) * np.abs(D[:, j])
            dr[:, j] = np.prod(ss[:, _b], axis=1) * dd
        return dr


class Exponential(Correlation):
    """Exponential (absolute-distance) kernel: ``exp(-sum_k theta_k * |D_k|)``."""

    def __call__(self, D, theta):
        return np.exp(np.sum(np.abs(D) * -theta, axis=1))

    def grad(self, D, theta):
        return -theta * np.sign(D) * self(D, theta)[:, None]

    def _dtheta_per_dim(self, D, theta):
        # exp(-sum theta_k |d_k|) factorizes, so d/d(theta_k) = -|d_k| * corr.
        return -np.abs(D) * self(D, theta)[:, None]


class Linear(Correlation):
    """Linear correlation kernel: ``prod_k max(1 - theta_k * |D_k|, 0)`` (compact support).

    The matching regression trend is named ``LinearRegression`` (in ``pydacefit.regr``),
    so this bare ``Linear`` kernel and that trend can be imported together without a clash.
    """

    def __call__(self, D, theta):
        return np.prod(np.maximum(1 - np.abs(D) * theta, 0), axis=1)

    def grad(self, D, theta):
        dr = np.zeros(D.shape)
        td = np.maximum(1 - np.abs(D) * theta, 0)

        for j in range(D.shape[1]):
            _b = index_except(D.shape[1], [j])
            _theta = theta[j] if type(theta) is np.ndarray and len(theta) == D.shape[1] else theta
            # (td_j > 0) zeroes the partial in the clamped (compact-support) tail, where
            # the factor is flat at 0 -- without it the derivative leaks a spurious -theta.
            dr[:, j] = np.prod(td[:, _b], axis=1) * -_theta * np.sign(D[:, j]) * (td[:, j] > 0)
        return dr

    def _dtheta_per_dim(self, D, theta):
        td = np.maximum(1 - np.abs(D) * theta, 0)
        # d s_j/d theta_j = -|D_j| where the factor is positive, 0 in the clamped region
        ds = -np.abs(D) * (td > 0)
        dr = np.zeros(D.shape)
        for j in range(D.shape[1]):
            _b = index_except(D.shape[1], [j])
            dr[:, j] = np.prod(td[:, _b], axis=1) * ds[:, j]
        return dr


class Spherical(Correlation):
    """Spherical kernel: compact-support correlation ``1 - 1.5 td + 0.5 td**3``."""

    def __call__(self, D, theta):
        td = np.minimum(np.abs(D) * theta, 1)
        ss = 1 - td * (1.5 - 0.5 * np.power(td, 2))
        r = np.prod(ss, axis=1)
        return r

    def grad(self, D, theta):
        dr = np.zeros(D.shape)
        td = np.minimum(np.abs(D) * theta, 1)
        ss = 1 - td * (1.5 - 0.5 * np.power(td, 2))

        for j in range(D.shape[1]):
            _theta = theta[j] if type(theta) is np.ndarray and len(theta) == D.shape[1] else theta
            dd = 1.5 * _theta * np.sign(D[:, j]) * (td[:, j] ** 2 - 1)
            _b = index_except(D.shape[1], [j])
            dr[:, j] = np.prod(ss[:, _b], axis=1) * dd
        return dr

    def _dtheta_per_dim(self, D, theta):
        td = np.minimum(np.abs(D) * theta, 1)
        ss = 1 - td * (1.5 - 0.5 * np.power(td, 2))
        dr = np.zeros(D.shape)
        for j in range(D.shape[1]):
            _b = index_except(D.shape[1], [j])
            # d s/d t = 1.5(t^2 - 1); d t/d theta = |D_j| (both 0 at the clamp t=1)
            dd = 1.5 * (td[:, j] ** 2 - 1) * np.abs(D[:, j])
            dr[:, j] = np.prod(ss[:, _b], axis=1) * dd
        return dr


class Spline(Correlation):
    """Cubic-spline kernel: piecewise-polynomial compact-support correlation."""

    def __call__(self, D, theta):
        ss = np.zeros(D.shape)
        xi = np.abs(D) * theta

        lo = xi <= 0.2
        mid = (xi > 0.2) & (xi < 1.0)
        ss[lo] = 1 - xi[lo] ** 2 * (15 - 30 * xi[lo])
        ss[mid] = 1.25 * (1 - xi[mid]) ** 3

        r = np.prod(ss, axis=1)
        return r

    def grad(self, D, theta):
        ss = np.zeros(D.shape)
        xi = np.abs(D) * theta
        lo = xi <= 0.2
        mid = (xi > 0.2) & (xi < 1.0)
        ss[lo] = 1 - xi[lo] ** 2 * (15 - 30 * xi[lo])
        ss[mid] = 1.25 * (1 - xi[mid]) ** 3

        dr = np.zeros(D.shape)
        n = D.shape[1]
        u = np.sign(D) * theta  # d(xi)/d(x) = sign(D) * theta

        # region masks key on the scaled distance xi (= |D|*theta), NOT on u.
        # d/d(xi) of the low branch is (90 xi - 30) xi; of 1.25(1-xi)^3 is -3.75(1-xi)^2;
        # chain by u for d/d(x).
        dr[lo] = u[lo] * ((90 * xi[lo] - 30) * xi[lo])
        dr[mid] = -3.75 * u[mid] * (1 - xi[mid]) ** 2

        for j in range(n):
            _ss = np.copy(ss)
            _ss[:, j] = dr[:, j]
            dr[:, j] = np.prod(_ss, axis=1)

        return dr

    def _dtheta_per_dim(self, D, theta):
        xi = np.abs(D) * theta
        lo = xi <= 0.2
        mid = (xi > 0.2) & (xi < 1.0)
        # per-dim correlation factor s_j and its derivative w.r.t. xi
        ss = np.zeros(D.shape)
        ss[lo] = 1 - xi[lo] ** 2 * (15 - 30 * xi[lo])
        ss[mid] = 1.25 * (1 - xi[mid]) ** 3
        dsdxi = np.zeros(D.shape)
        dsdxi[lo] = -30 * xi[lo] + 90 * xi[lo] ** 2
        dsdxi[mid] = -3.75 * (1 - xi[mid]) ** 2
        # d s_j/d theta_j = (d s/d xi)(d xi/d theta) with d xi/d theta = |D_j|
        ds = dsdxi * np.abs(D)
        dr = np.zeros(D.shape)
        for j in range(D.shape[1]):
            _b = index_except(D.shape[1], [j])
            dr[:, j] = np.prod(ss[:, _b], axis=1) * ds[:, j]
        return dr


class GeneralizedExponential(Correlation):
    """Generalized exponential kernel: ``exp(-theta * |D|**power)``.

    Generalizes the fixed-exponent kernels by making the exponent a parameter:
    ``power = 2`` recovers ``Gaussian`` and ``power = 1`` recovers ``Exponential``, so
    ``1 <= power <= 2`` interpolates between them. Unlike the other kernels the
    exponent is tuned *with* the length-scale: ``theta`` is ``(length_scale, power)``
    (or ``(length_scales..., power)`` for ARD), so the search optimizes both together
    -- which is why ``power`` stays in ``theta`` here rather than on the object.
    """

    def __call__(self, D, theta):
        # theta is (length_scale, power) [isotropic] or (length_scales..., power) [ARD];
        # D is the (n_pairs, d) difference matrix, so the ARD case is keyed on the input
        # dimensionality D.shape[1], NOT len(D) (which is the pair count).
        _theta, power = self._split(D, theta)
        return np.exp(np.sum(np.abs(D) ** power * -_theta, axis=1))

    def grad(self, D, theta):
        _theta, power = self._split(D, theta)
        return power * -_theta * np.sign(D) * np.abs(D) ** (power - 1) * self(D, theta)[:, None]

    def theta_grad(self, D, theta):
        # theta = (length_scale(s)..., power). One column per entry: the length-scale
        # partials first, then a final column for the shared exponent ``power``. This
        # kernel overrides theta_grad (rather than _dtheta_per_dim) because of that
        # extra power column and the special theta layout.
        _theta, power = self._split(D, theta)
        corr = self(D, theta)[:, None]
        ad = np.abs(D)
        adp = ad**power

        # length-scale partials: d corr / d theta_k = -|D_k|^power * corr
        d_ls = -adp * corr
        if np.size(_theta) == 1:  # isotropic: one shared length-scale -> a single column
            d_ls = d_ls.sum(axis=1, keepdims=True)

        # power partial: d corr / d power = -corr * sum_k theta_k |D_k|^power ln|D_k|
        # (the |D_k|=0 terms vanish: x^p ln x -> 0, but numerically 0 * -inf -> nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            plog = adp * np.log(ad)
        plog = np.where(ad > 0, plog, 0.0)
        d_pow = -np.sum(_theta * plog, axis=1, keepdims=True) * corr

        return np.concatenate([d_ls, d_pow], axis=1)

    @staticmethod
    def _split(D, theta):
        """Split theta into its length-scale(s) and the shared exponent ``power``."""
        d = D.shape[1]
        if len(theta) == 2:
            return theta[0], theta[1]
        if len(theta) == d + 1:
            return theta[:-1], theta[-1]
        raise Exception(f"For GeneralizedExponential theta is either length 2 or d+1 = {d + 1}")


class RationalQuadratic(Correlation):
    """Rational Quadratic correlation kernel: an (infinite) scale-mixture of Gaussians.

    The shape parameter ``alpha`` sets the tails -- small alpha gives heavier tails
    (more robust across length scales), and ``alpha -> inf`` recovers ``Gaussian``.
    It is a fixed construction parameter, deliberately kept out of ``theta`` (which
    the optimizer tunes by maximum likelihood): tuning a shape parameter on a small
    sample overfits, so it belongs to the kernel, not to the search vector.

    The default ``alpha=0.25`` gives noticeably heavy tails (robust across a range of
    length scales -- a safe general-purpose choice); raise it toward ``Gaussian`` for a
    tighter, single-scale fit. ``theta`` carries the length-scale(s) and may be a scalar
    (isotropic) or a per-dimension vector (ARD), like the other kernels.

    Args:
        alpha: Tail / scale-mixture parameter (> 0).
    """

    def __init__(self, alpha=0.25):
        self.alpha = alpha

    def __call__(self, D, theta):
        # parameterized so alpha -> inf recovers Gaussian exactly (no 1/2 factor,
        # matching this library's gauss convention exp(-theta * D**2)).
        base = 1 + theta * np.square(D) / self.alpha
        return np.prod(base ** (-self.alpha), axis=1)

    def grad(self, D, theta):
        base = 1 + theta * np.square(D) / self.alpha
        r = np.prod(base ** (-self.alpha), axis=1)
        return -2 * theta * D / base * r[:, None]

    def _dtheta_per_dim(self, D, theta):
        # the k-th factor's log-derivative is -d_k^2 / base_k (the alpha's cancel),
        # so d corr / d theta_k = -d_k^2 / base_k * corr, per input dimension.
        base = 1 + theta * np.square(D) / self.alpha
        r = np.prod(base ** (-self.alpha), axis=1)[:, None]
        return -np.square(D) / base * r

    def __repr__(self):
        return f"RationalQuadratic(alpha={self.alpha})"


class Matern(Correlation):
    """Matérn kernel (product of per-dimension Matérns), smoothness ``nu`` in {0.5, 1.5, 2.5}.

    ``nu`` controls smoothness: 0.5 is the rough exponential (once-continuous), 2.5 is
    twice-differentiable (a common default for physical responses, where the Gaussian's
    infinite smoothness is unrealistic), and ``nu -> inf`` recovers ``Gaussian``. Like
    RQ's ``alpha``, ``nu`` is a fixed construction parameter kept out of ``theta`` --
    tuning smoothness on a small sample overfits. ``theta`` carries the inverse
    length-scale(s), scalar (isotropic) or per-dimension (ARD); the kernel is the product
    of per-dimension Matérns ``M(theta_k * |D_k|)`` (the standard half-integer closed
    forms, so theta=1/length recovers textbook Matérn-nu).

    Args:
        nu: Smoothness; one of 0.5, 1.5, 2.5 (the cases with a closed form).
    """

    def __init__(self, nu=2.5):
        if nu not in (0.5, 1.5, 2.5):
            raise ValueError("Matern supports nu in {0.5, 1.5, 2.5} (the closed-form cases).")
        self.nu = nu

    def _factor(self, D, theta):
        # per-dimension scaled distance s = theta_k |D_k|, the Matérn factor M(s) and its
        # derivative M'(s) -- both (n_pairs, d). These build the product kernel and both
        # gradients, exactly like the other product kernels (Cubic/Spherical/Spline).
        s = np.abs(D) * theta
        if self.nu == 0.5:
            e = np.exp(-s)
            return e, -e
        if self.nu == 1.5:
            c = np.sqrt(3.0)
            e = np.exp(-c * s)
            return (1 + c * s) * e, -3.0 * s * e
        # nu == 2.5
        c = np.sqrt(5.0)
        e = np.exp(-c * s)
        return (1 + c * s + (5.0 / 3.0) * s**2) * e, -(5.0 / 3.0) * s * (1 + c * s) * e

    def __call__(self, D, theta):
        ss, _ = self._factor(D, theta)
        return np.prod(ss, axis=1)

    def grad(self, D, theta):
        ss, ds = self._factor(D, theta)
        dr = np.zeros(D.shape)
        for j in range(D.shape[1]):
            _b = index_except(D.shape[1], [j])
            _theta = theta[j] if type(theta) is np.ndarray and len(theta) == D.shape[1] else theta
            # dK/dD_j = prod_{k!=j} M_k * M'(s_j) * ds_j/dD_j, with ds_j/dD_j = theta_j sign(D_j)
            dr[:, j] = np.prod(ss[:, _b], axis=1) * ds[:, j] * _theta * np.sign(D[:, j])
        return dr

    def _dtheta_per_dim(self, D, theta):
        ss, ds = self._factor(D, theta)
        dr = np.zeros(D.shape)
        for j in range(D.shape[1]):
            _b = index_except(D.shape[1], [j])
            # dK/d theta_j = prod_{k!=j} M_k * M'(s_j) * ds_j/d theta_j, with ds_j/d theta_j = |D_j|
            dr[:, j] = np.prod(ss[:, _b], axis=1) * ds[:, j] * np.abs(D[:, j])
        return dr

    def __repr__(self):
        return f"Matern(nu={self.nu})"


def index_except(n, indices):
    return [i for i in range(n) if i not in indices]
