import numpy as np


def regr_constant(X):
    return np.ones((X.shape[0], 1))


def regr_constant_grad(X):
    return np.zeros((X.shape[1], 1))


def regr_linear(X):
    return np.column_stack([np.ones((X.shape[0], 1)), X])


def regr_linear_grad(X):
    return np.column_stack([np.zeros((X.shape[1], 1)), np.eye(X.shape[1])])


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


def regr_quadratic_grad(X):
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
