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


def corr_spline(D, theta):
    ss = np.zeros(D.shape)
    xi = np.abs(D) * theta

    I = np.where(xi <= 0.2)
    if len(I) > 0:
        ss[I] = 1 - xi[I] ** 2 * (15 - 30 * xi[I])

    I = np.where(np.logical_and(xi > 0.2, xi < 1.0))
    if len(I) > 0:
        ss[I] = 1.25 * (1 - xi[I]) ** 3

    r = np.prod(ss, axis=1)
    return r


def corr_expg(D, theta):
    if len(theta) == 2:
        _theta = theta[0]
        power = theta[1]
    elif len(theta) == len(D) + 1:
        _theta = theta[:-1]
        power = theta[-1]
    else:
        raise Exception("For corr_expg theta is either length of 2 or D+1 = %s " % (len(D) + 1))

    return np.exp(np.sum(np.abs(D) ** power * -_theta, axis=1))
