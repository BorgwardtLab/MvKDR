# MvKDR
This repository contains Python implementations of the algorithm MvKDR described in "Multi-view Spectral Clustering on Conflicting Views", which is appearing in ECML/PKDD 2017, SKOPJE, MACEDONIA 

## Dependencies
**Python 2.7**, modern versions of **numpy**, **scipy**, **pandas**, **scikit-learn**. All of them available via **pip**.

## Usage
The implementation of MvKDR is in code/mvkdr.py (currently only two views are supported)

    Km_label, km_obj = mvkdr(X1, X2, sigma1, sigma2, lambda1, lambda2, seed)
    
input:

    X1: a n X p1 numpy matrix of n samples and p1 feaures in view 1

    X2: a n X p2 numpy matrix of n samples and p2 features in view 2

    sigma1:  a float for sigma for gaussian kernel for X1, should be set to the median of pairwise distance of X1
    
    sigma2:  a float for sigma for gaussian kernel for X2, should be set to the median of pairwise distance of X2
    
    lambda1:  a float indictes the regularization parameter of agreement between subspace projection 
    
    lambda2:  a float indictes the regularization parameter of disagreement between alternative subspace projection
    
    seed:   an integer indicates the seed for initialization

Output:

    km_label:  a vector of size n for clustering label produced by k-means
    
    km_obj: a float of k-means objective value
   
## Contact
Any questions can be directed to:
   * Xiao He: xiao.he [at] bsse.ethz.ch
