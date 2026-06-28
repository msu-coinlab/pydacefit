"""Regression (trend) basis functions for the DACE model, as first-class objects."""

import numpy as np


class Regression:
    """A regression trend: a callable object bundling its own gradient.

    - ``__call__(X)`` -- the design matrix ``F`` of basis functions evaluated at ``X``,
      shape ``(m, p)`` for ``m`` rows and ``p`` basis terms.
    - ``grad(X)`` -- derivative of the basis w.r.t. the design point, shape ``(m, d, p)``
      (one ``(d, p)`` Jacobian per row), used by ``predict(grad=True)``.

    Both are vectorized over the rows of ``X`` so ``predict`` can evaluate every query
    point in one batched call. ``grad`` is optional in the same sense as
    ``Correlation.grad``: a trend without an analytic form leaves it raising
    ``NotImplementedError``.
    """

    def __call__(self, X):
        raise NotImplementedError

    def grad(self, X):
        raise NotImplementedError

    def __repr__(self):
        return type(self).__name__


class ConstantRegression(Regression):
    """Constant trend: a single intercept column of ones."""

    def __call__(self, X):
        return np.ones((X.shape[0], 1))

    def grad(self, X):
        return np.zeros((X.shape[0], X.shape[1], 1))


class LinearRegression(Regression):
    """Linear trend: intercept plus one column per input dimension."""

    def __call__(self, X):
        return np.column_stack([np.ones((X.shape[0], 1)), X])

    def grad(self, X):
        m, d = X.shape
        # per row the Jacobian is [d(1)/dx=0 | d(x)/dx=I]; identical for every row.
        g = np.zeros((m, d, 1 + d))
        g[:, :, 1:] = np.eye(d)
        return g


class QuadraticRegression(Regression):
    """Quadratic trend: intercept, linear terms and all pairwise products."""

    @staticmethod
    def _pairs(d):
        # column order of the quadratic block: (a, b) with a <= b, matching __call__.
        return [(a, b) for a in range(d) for b in range(a, d)]

    def __call__(self, X):
        m, n, nn = X.shape[0], X.shape[1], int((X.shape[1] + 1) * (X.shape[1]) / 2)

        M = np.zeros((m, nn))

        j = 0
        q = n

        for k in range(n):
            M[:, j + np.arange(q)] = X[:, [k]] * X[:, np.arange(k, n)]
            j += q
            q -= 1

        return np.column_stack([np.ones((X.shape[0], 1)), X, M])

    def grad(self, X):
        m, d = X.shape
        # intercept (zero) and linear (identity) blocks, broadcast over rows
        inter = np.zeros((m, d, 1))
        lin = np.broadcast_to(np.eye(d), (m, d, d))
        # quadratic block: d(x_a x_b)/dx_j = x_b[j=a] + x_a[j=b] (2 x_a when a == b)
        pairs = self._pairs(d)
        quad = np.zeros((m, d, len(pairs)))
        for c, (a, b) in enumerate(pairs):
            if a == b:
                quad[:, a, c] = 2 * X[:, a]
            else:
                quad[:, a, c] = X[:, b]
                quad[:, b, c] = X[:, a]
        return np.concatenate([inter, lin, quad], axis=2)
