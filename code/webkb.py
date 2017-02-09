import numpy as np
import scipy.io as sio

from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

def load_single(name):
    data = sio.loadmat('../data/webkb/%s_2V_Full.mat' % name)
    X = data['X']
    y = data['y']
    id = data['id']
    X1 = X[id[0][0][0]-1, :].T
    X2 = X[id[0][1][0]-1, :].T
    y = y.reshape([y.shape[0],]) - 1
    X1 = X1.astype(float)
    X2 = X2.astype(float)
    X1 = X1[:,np.where(X1.any(axis=0))[0]]
    X2 = X2[:,np.where(X2.any(axis=0))[0]]

    svd = TruncatedSVD(100, algorithm='arpack')
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    X1 = lsa.fit_transform(X1)
    X2 = lsa.fit_transform(X2)

    return X1, X2, y    

def load_all():
    names = ['Washington', 'Cornell', 'Texas', 'Wisconsin']
    for i, name in enumerate(names):
        data = sio.loadmat('../data/webkb/%s_2V_Full.mat' % name)
        X = data['X']
        y = data['y']
        id = data['id']
        X1 = X[id[0][0][0]-1, :].T
        X2 = X[id[0][1][0]-1, :].T
        y = y.reshape([y.shape[0],]) - 1
        X1 = X1.astype(float)
        X2 = X2.astype(float)

        if i == 0:
            V1 = X1
            V2 = X2
            Y = y
        else:
            V1 = np.vstack([V1, X1])
            X2shape = X2.shape[0]
            V2shape = V2.shape[0]
            V2 = np.hstack([V2, np.zeros([V2shape, X2shape])])
            X2 = np.hstack([np.zeros([X2shape, V2shape]), X2])
            V2 = np.vstack([V2, X2])
            Y = np.append(Y, y)
            
    X1, X2, y = V1, V2, Y
    X1 = X1[:,np.where(X1.any(axis=0))[0]]
    X2 = X2[:,np.where(X2.any(axis=0))[0]]

    svd = TruncatedSVD(100, algorithm='arpack')
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    X1 = lsa.fit_transform(V1)
    X2 = lsa.fit_transform(V2)

    return X1, X2, y