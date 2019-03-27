import autograd.numpy as np


# function to calculate the correlation matrix all in one
def calc_kernel_matrix(A, B, func, theta):
    D = np.repeat(A, B.shape[0], axis=0) - np.tile(B, (A.shape[0], 1))
    K = func(D, theta)
    return np.reshape(K, (A.shape[0], B.shape[0]))


def calc_grad(A, B, func, theta):
    D = np.repeat(A, B.shape[0], axis=0) - np.tile(B, (A.shape[0], 1))
    return func(D, theta)


# -------------------------------
# Correlation Functions
# -------------------------------

def corr_gauss(D, theta):
    return np.exp(np.sum(np.square(D) * -theta, axis=1))


def corr_gauss_grad(D, theta):
    return -2 * theta * D * corr_gauss(D, theta)[:, None]


def corr_cubic(D, theta):
    td = np.minimum(np.abs(D) * theta, 1)
    ss = 1 - td ** 2 * (3 - 2 * td)
    r = np.prod(ss, axis=1)
    return r


def corr_cubic_grad(D, theta):
    dr = np.zeros(D.shape)
    td = np.minimum(np.abs(D) * theta, 1)
    ss = 1 - td ** 2 * (3 - 2 * td)

    for j in range(D.shape[1]):
        _b = index_except(D.shape[1], [j])
        _theta = theta[j] if type(theta) == np.ndarray and len(theta) == D.shape[1] else theta
        dd = 6 * _theta * np.sign(D[:, j]) * td[:, j] * (td[:, j] - 1)

        dr[:, j] = np.prod(ss[:, _b], axis=1) * dd
    return dr


def corr_exp(D, theta):
    return np.exp(np.sum(np.abs(D) * -theta, axis=1))


def corr_exp_grad(D, theta):
    return - theta * np.sign(D) * corr_exp(D, theta)[:, None]


def corr_lin(D, theta):
    return np.prod(np.maximum(1 - np.abs(D) * theta, 0), axis=1)


def corr_lin_grad(D, theta):
    dr = np.zeros(D.shape)
    td = np.maximum(1 - np.abs(D) * theta, 0)

    for j in range(D.shape[1]):
        _b = index_except(D.shape[1], [j])
        _theta = theta[j] if type(theta) == np.ndarray and len(theta) == D.shape[1] else theta
        dr[:, j] = np.prod(td[:, _b], axis=1) * -_theta * np.sign(D[:, j])
    return dr


def corr_spherical(D, theta):
    td = np.minimum(np.abs(D) * theta, 1)
    ss = 1 - td * (1.5 - 0.5 * np.power(td, 2))
    r = np.prod(ss, axis=1)
    return r


def corr_spherical_grad(D, theta):
    dr = np.zeros(D.shape)
    td = np.minimum(np.abs(D) * theta, 1)
    ss = 1 - td * (1.5 - 0.5 * np.power(td, 2))

    for j in range(D.shape[1]):
        _theta = theta[j] if type(theta) == np.ndarray and len(theta) == D.shape[1] else theta
        dd = 1.5 * _theta * np.sign(D[:, j]) * (td[:, j] ** 2 - 1)
        _b = index_except(D.shape[1], [j])
        dr[:, j] = np.prod(ss[:, _b], axis=1) * dd
    return dr


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


def corr_spline_grad(D, theta):
    ss = np.zeros(D.shape)
    xi = np.abs(D) * theta
    I = np.where(xi <= 0.2)
    if len(I) > 0:
        ss[I] = 1 - xi[I] ** 2 * (15 - 30 * xi[I])
    I = np.where(np.logical_and(xi > 0.2, xi < 1.0))
    if len(I) > 0:
        ss[I] = 1.25 * (1 - xi[I]) ** 3

    dr = np.zeros(D.shape)
    m, n = D.shape
    u = np.sign(D) * theta

    I = np.where(u <= 0.2)
    if len(I) > 0:
        dr[I] = u[I] * ((90 * xi[I] - 30) * xi[I])
    I = np.where(np.logical_and(xi > 0.2, xi < 1.0))
    if len(I) > 0:
        dr[I] = -3.75 * u[I] * (1 - xi[I] ** 2)

    for j in range(n):
        _ss = np.copy(ss)
        _ss[:, j] = dr[:, j]
        dr[:, j] = np.prod(_ss, axis=1)

    return dr


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


def corr_expg_grad(D, theta):
    if len(theta) == 2:
        _theta = theta[0]
        power = theta[1]
    elif len(theta) == len(D) + 1:
        _theta = theta[:-1]
        power = theta[-1]

    return power * -_theta * np.sign(D) * np.abs(D) ** (power - 1) * corr_expg(D, theta)[:, None]


def index_except(n, indices):
    return [i for i in range(n) if i not in indices]
