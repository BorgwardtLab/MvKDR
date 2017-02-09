import numpy as np
import numpy.matlib as matlib
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import IPython as ip

from utility import laplace, eigh, centering
from kdr import kdr

TH       = 1e-4
MAX_LOOP = 50
VERBOSE  = 0

def mvkdr(X1, X2, k, sigma1, sigma2, theta1, theta2, seed):
    n = X1.shape[0]
    unit = np.ones([n, n])
    I = np.eye(n)
    Q = I - unit/n      
    K1, D1, L1 = laplace(X1, sigma1)
    K2, D2, L2 = laplace(X2, sigma2)    
    K3, D3, L3 = laplace(X2, sigma2)
    U1 = eigh(L1, k, 'descend')
    U2 = eigh(L2, k, 'descend')
    U3 = U2  
    G1 = np.dot(np.dot(np.dot(D1, U1), U1.T), D1)
    G1 = np.dot(np.dot(Q, G1), Q)   
    G2 = np.dot(np.dot(np.dot(D2, U2), U2.T), D2)
    G2 = np.dot(np.dot(Q, G2), Q)   
    G3 = np.dot(np.dot(np.dot(D3, U3), U3.T), D3)
    G3 = np.dot(np.dot(Q, G3), Q)
    W1 = kdr(X1, G1, None, k-1, sigma1, seed)
    W2 = kdr(X2, G2, None, k-1, sigma2, seed)       
    W3 = W2
    obj = np.zeros([MAX_LOOP,])
    for h in range(MAX_LOOP):
        K1, D1, L1 = laplace(np.dot(X1, W1), sigma1)
        K2, D2, L2 = laplace(np.dot(X2, W2), sigma2)        
        K3, D3, L3 = laplace(np.dot(X2, W3), sigma2)
        for i in range(1):
            U2 = eigh(L2 + theta1*np.dot(U1, U1.T) - theta2*np.dot(U3, U3.T), k, 'descend')
            U3 = eigh(L3 - theta2*np.dot(U1, U1.T) - theta2*np.dot(U2, U2.T), k, 'descend')
            U1 = eigh(L1 + theta1*np.dot(U2, U2.T) - theta2*np.dot(U3, U3.T), k, 'descend')
        UU1 = np.dot(U1, U1.T)
        UU2 = np.dot(U2, U2.T)
        UU3 = np.dot(U3, U3.T)
        obj[h] = np.sum(UU1*L1) + np.sum(UU2*L2) + np.sum(UU3*L3) + theta1*np.sum(UU1*UU2) - theta2*np.sum(UU1*UU3) - theta2*np.sum(UU2*UU3)
        if VERBOSE == 1:
            print 'mvkdr %s obj: %s' % (h, obj[h])
        if h > 0 and abs(obj[h] - obj[h-1]) < TH:
            break
        G1 = np.dot(np.dot(np.dot(D1, U1), U1.T), D1)
        G1 = np.dot(np.dot(Q, G1), Q)           
        G2 = np.dot(np.dot(np.dot(D2, U2), U2.T), D2)
        G2 = np.dot(np.dot(Q, G2), Q)   
        G3 = np.dot(np.dot(np.dot(D3, U3), U3.T), D3)
        G3 = np.dot(np.dot(Q, G3), Q)
        W1 = kdr(X1, G1, W1, k-1, sigma1, seed)     
        W2 = kdr(X2, G2, W2, k-1, sigma2, seed)     
        W3 = kdr(X2, G3, W3, k-1, sigma2, seed) 
    km = KMeans(n_clusters=k, random_state=seed)
    U1 = normalize(U1)
    km.fit(U1)
    return km.labels_, km.inertia_
    