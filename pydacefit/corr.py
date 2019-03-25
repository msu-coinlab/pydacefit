import numpy as np


# function to calculate the correlation matrix all in one
def calc_kernel_matrix(A, B, func, theta):
    D = np.repeat(A, B.shape[0], axis=0) - np.tile(B, (A.shape[0], 1))
    K = func(D, theta)
    return np.reshape(K, (A.shape[0], B.shape[0]))


# -------------------------------
# Correlation Functions
# -------------------------------

def corr_gauss(D, theta):
    return np.exp(np.sum(np.square(D) * -theta, axis=1))


def corr_cubic(D, theta):
    td = np.minimum(np.abs(D) * theta, 1)
    ss = 1 - td ** 2 * (3 - 2 * td)
    r = np.prod(ss, axis=1)
    return r


def corr_exp(D, theta):
    return np.exp(np.sum(np.abs(D) * -theta, axis=1))


def corr_lin(D, theta):
    return np.prod(np.maximum(1 - np.abs(D) * theta, 0), axis=1)


def corr_spherical(D, theta):
    td = np.minimum(np.abs(D) * theta, 1)
    ss = 1 - td * (1.5 - 0.5 * np.power(td, 2))
    r = np.prod(ss, axis=1)
    return r
