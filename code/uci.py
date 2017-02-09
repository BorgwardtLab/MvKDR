import numpy as np
import pandas as pd
import IPython as ip

from sklearn.preprocessing import scale, LabelEncoder, Imputer

def iris(fn):
    X = np.loadtxt(fn, delimiter=',', dtype='str')
    y = X[:, 4]
    X = np.delete(X, 4, 1)
    X = np.vstack([X[y=='Iris-versicolor'], X[y=='Iris-virginica']])
    y = np.append(y[y=='Iris-versicolor'], y[y=='Iris-virginica'])

    X = X.astype(float)
    y = LabelEncoder().fit_transform(y)

    a = X.shape[1]/2
    l = np.zeros(X.shape[1])
    l[:a] = 0
    l[a:] = 1
    X1 = X[:, np.where(l==1)[0]]
    X2 = X[:, np.where(l==0)[0]]    

    X1 = scale(X1)
    X2 = scale(X2)

    return X1, X2, y

def glass(fn):
    X = np.loadtxt(fn, delimiter=',')
    y = X[:, 10]
    X = np.delete(X, 10, 1)
    X = np.delete(X, 0, 1)
    y[y==1] = 1
    y[y==2] = 1
    y[y==3] = 2
    y[y==5] = 0
    y[y==6] = 0
    y[y==7] = 0

    a = X.shape[1]/2
    l = np.zeros(X.shape[1])
    l[:a] = 1
    l[a:] = 0
    X1 = X[:, np.where(l==1)[0]]
    X2 = X[:, np.where(l==0)[0]]

    X1 = scale(X1)
    X2 = scale(X2)

    return X1, X2, y

def hepatitis(fn):
    X = pd.read_csv(fn, header=None, index_col=False)
    X = X.as_matrix()
    X = Imputer().fit_transform(X)
    y = X[:, 0] 
    X = np.delete(X, 0, 1)

    a = X.shape[1]/2
    l = np.zeros(X.shape[1])
    l[:a] = 0
    l[a:] = 1
    X1 = X[:, np.where(l==1)[0]]
    X2 = X[:, np.where(l==0)[0]]    

    X1 = scale(X1)
    X2 = scale(X2)

    return X1, X2, y

def ionosphere(fn):
    X = np.loadtxt(fn, delimiter=',', dtype='str')
    y = X[:, X.shape[1]-1]
    X = np.delete(X, [1, X.shape[1]-1], 1)  

    X = X.astype(float)
    y = LabelEncoder().fit_transform(y)

    a = X.shape[1]/2
    l = np.zeros(X.shape[1])
    l[:a] = 1
    l[a:] = 0
    X1 = X[:, np.where(l==1)[0]]
    X2 = X[:, np.where(l==0)[0]]    

    X1 = scale(X1)
    X2 = scale(X2)

    return X1, X2, y

def wine(fn):
    X = np.loadtxt(fn, delimiter=',')
    y = X[:, 0]
    X = np.delete(X, 0, 1)

    a = X.shape[1]/2
    l = np.zeros(X.shape[1])
    l[:a] = 0
    l[a:] = 1
    X1 = X[:, np.where(l==1)[0]]
    X2 = X[:, np.where(l==0)[0]]    

    X1 = scale(X1)
    X2 = scale(X2)

    return X1, X2, y

def wdbc(fn):
    X = np.loadtxt(fn, delimiter=',', dtype='str')
    y = X[:, 1]
    X = np.delete(X, [0, 1], 1)

    X = X.astype(float)
    y = LabelEncoder().fit_transform(y)

    a = X.shape[1]/2
    l = np.zeros(X.shape[1])
    l[:a] = 0
    l[a:] = 1
    X1 = X[:, np.where(l==1)[0]]
    X2 = X[:, np.where(l==0)[0]]    

    X1 = scale(X1)
    X2 = scale(X2)

    return X1, X2, y    