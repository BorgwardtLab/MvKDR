"""
Kernel dimensionality reduction

Xiao August 2016
"""

import numpy as np
import numpy.matlib as matlib

from scipy import linalg
from scipy import optimize

from utility import rbf, centering

MAX_LOOP = 50
TH       = 1e-4
ETA      = 0.1
VERBOSE  = 0

def kdr1dim(s, X, G, W, dW, sigma):
    tempW = W+s*dW
    tempW = linalg.svd(tempW, full_matrices=False)[0]
    K = rbf(np.dot(X, tempW), sigma=sigma)
    return - np.sum(G*K)

def kdr(X, G, W, k, sigma, seed):
    n, p = X.shape
    np.random.seed(seed)
    if W is None:
        W = np.random.rand(p, k)
        W = linalg.svd(W, full_matrices=False)[0]
    dW = np.zeros([p, k])   
    obj = np.zeros([MAX_LOOP+1,])
    K = rbf(np.dot(X, W), sigma)
    obj[0] = np.sum(G*K)
    XX = np.zeros([p, n*n])
    ZZ = np.zeros([k, n*n])
    for i in range(p):
        Xi = matlib.repmat(X[:, i], n, 1).T
        XX[i] = (Xi - Xi.T).reshape([n*n,]) 
    for h in range(MAX_LOOP):
        Z = np.dot(X, W)
        for j in range(k):
            Zj = matlib.repmat(Z[:, j], n, 1).T
            ZZ[j] = (Zj - Zj.T).reshape([n*n,])
        dW = - np.dot(G.reshape([n*n,]) * K.reshape([n*n,]) * XX, ZZ.T)/sigma/sigma
        dW = dW / linalg.norm(dW)
        s = optimize.fminbound(kdr1dim, 0, ETA, args=(X, G, W, dW, sigma))
        W = W+s*dW
        W = linalg.svd(W, full_matrices=False)[0]
        K = rbf(np.dot(X, W), sigma)
        obj[h+1] = np.sum(G*K)
        if VERBOSE == 1:
            print 'kdr obj: %s, %s' % (obj[h+1], s)
        if (abs(obj[h+1] - obj[h]) < TH):
            break
    if VERBOSE == 1:
        print
    return W
