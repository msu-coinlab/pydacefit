import numpy as np


def regr_constant(X, return_gradient=False):
    return np.ones((X.shape[0], 1))

def regr_linear(X):
    return np.column_stack([np.ones((X.shape[0], 1)), X])


def regr_quadratic(X):
    m, n, nn = X.shape[0], X.shape[1], int((X.shape[1] + 1) * (X.shape[1]) / 2)

    M = np.zeros((m, nn))

    j = 0
    q = n

    for k in range(n):
        M[:, j + np.arange(q)] = X[:, [k]] * X[:, np.arange(k, n)]
        j += q
        q -= 1

    return np.column_stack([np.ones((X.shape[0], 1)), X, M])
