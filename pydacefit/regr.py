"""Regression (trend) basis functions for the DACE model, as first-class objects."""

import numpy as np


class Regression:
    """A regression trend: a callable object bundling its own gradient.

    - ``__call__(X)`` -- the design matrix ``F`` of basis functions evaluated at ``X``.
    - ``grad(X)`` -- derivative of the basis w.r.t. the design point, used by
      ``predict(grad=True)``.

    ``grad`` is optional in the same sense as ``Correlation.grad``: a trend without an
    analytic form leaves it raising ``NotImplementedError``.
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
        return np.zeros((X.shape[1], 1))


class LinearRegression(Regression):
    """Linear trend: intercept plus one column per input dimension."""

    def __call__(self, X):
        return np.column_stack([np.ones((X.shape[0], 1)), X])

    def grad(self, X):
        return np.column_stack([np.zeros((X.shape[1], 1)), np.eye(X.shape[1])])


class QuadraticRegression(Regression):
    """Quadratic trend: intercept, linear terms and all pairwise products."""

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
        m, n = X.shape
        nn = int((n + 1) * (n + 2) / 2)

        df = np.column_stack([np.zeros((n, 1)), np.eye(n), np.zeros((n, nn - n - 1))])

        q = n
        j = n + 1

        for k in range(n):
            df[k, j + np.arange(q)] = np.column_stack([2 * X[:, [k]], X[:, np.arange(k + 1, n)]])

            for i in range(n - k - 1):
                df[k + i + 1, j + 1 + i] = X[0, k]
            j = j + q
            q = q - 1

        return df
