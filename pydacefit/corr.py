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
