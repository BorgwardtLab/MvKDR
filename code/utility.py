import math
import numpy as np
from scipy import linalg


def centering(K):
    n = K.shape[0]
    unit = np.ones([n, n])
    I = np.eye(n)
    Q = I - unit/n
    
    return np.dot(np.dot(Q, K), Q)


def mediandist(X):
    GX = np.dot(X, X.T)
    KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
    mdist = np.median(KX[KX != 0])
    return mdist


def rbf(X, sigma=None):
    GX = np.dot(X, X.T)
    KX = np.diag(GX) - GX + (np.diag(GX) - GX).T
    if sigma is None:
        mdist = np.median(KX[KX != 0])
        sigma = math.sqrt(mdist)
    KX *= - 0.5 / sigma / sigma
    np.exp(KX, KX)
    return KX


def laplace(X, sigma=None):
    K = rbf(X, sigma)
    np.fill_diagonal(K, 0)
    D_sq = np.diag(1.0 / np.sqrt(np.sum(K, axis=0)))
    L = np.dot(np.dot(D_sq, K), D_sq)
    return K, D_sq, L    


def eigh(K, k, order):
    n = K.shape[0]
    if order == 'descend':
        w, U = linalg.eigh(K, eigvals=[n - k, n - 1])
    elif order == 'ascend':
        w, U = linalg.eigh(K, eigvals=[0, k-1])
    else:
        exit()
    return U