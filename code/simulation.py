import numpy as np

def simulation(seed, alpha, beta):
    np.random.seed(seed)

    N = 100

    clus1_mean_view1 = [1, 1]
    clus2_mean_view1 = [3, 3]
    clus3_mean_view1 = [4, -1]
    clus1_cov_view1 = [[1, 0], [0, 1]]
    clus2_cov_view1 = [[1, 0], [0, 1]]
    clus3_cov_view1 = [[1, 0], [0, 1]]        
    clus1_view1 = np.random.multivariate_normal(clus1_mean_view1, clus1_cov_view1, N)
    clus2_view1 = np.random.multivariate_normal(clus2_mean_view1, clus2_cov_view1, N)
    clus3_view1 = np.random.multivariate_normal(clus3_mean_view1, clus3_cov_view1, N)
    view1 = np.append(clus1_view1, clus2_view1, axis=0)
    view1 = np.append(view1, clus3_view1, axis=0)    

    clus1_mean_view2 = [1, 1]
    clus2_mean_view2 = [3, 3]
    clus3_mean_view2 = [4, -1]    
    clus1_cov_view2 = [[1, 0], [0, 1]]
    clus2_cov_view2 = [[1, 0], [0, 1]]
    clus3_cov_view2 = [[1, 0], [0, 1]]    
    clus1_view2 = np.random.multivariate_normal(clus1_mean_view2, clus1_cov_view2, N)
    clus2_view2 = np.random.multivariate_normal(clus2_mean_view2, clus2_cov_view2, N)
    clus3_view2 = np.random.multivariate_normal(clus3_mean_view2, clus3_cov_view2, N)
    view2 = np.append(clus1_view2, clus2_view2, axis=0)
    view2 = np.append(view2, clus3_view2, axis=0)

    y1 = np.zeros([view1.shape[0],], int)
    y1[:N] = 1
    y1[N:2*N] = 2        

    clus1_mean_view3 = [1, 1]
    clus2_mean_view3 = [3, 3]
    clus3_mean_view3 = [4, -1]
    clus1_cov_view3 = [[1, 0], [0, 1]]
    clus2_cov_view3 = [[1, 0], [0, 1]]
    clus3_cov_view3 = [[1, 0], [0, 1]]
    clus1_view3 = np.random.multivariate_normal(clus1_mean_view3, clus1_cov_view3, N)
    clus2_view3 = np.random.multivariate_normal(clus2_mean_view3, clus2_cov_view3, N)
    clus3_view3 = np.random.multivariate_normal(clus3_mean_view3, clus3_cov_view3, N)
    view3 = np.append(clus1_view3, clus2_view3, axis=0)
    view3 = np.append(view3, clus3_view3, axis=0)    

    clus1_mean_view4 = [1, 1]
    clus2_mean_view4 = [3, 3]
    clus3_mean_view4 = [4, -1]
    clus1_cov_view4 = [[1, 0], [0, 1]]
    clus2_cov_view4 = [[1, 0], [0, 1]]
    clus3_cov_view4 = [[1, 0], [0, 1]]
    clus1_view4 = np.random.multivariate_normal(clus1_mean_view4, clus1_cov_view4, N)
    clus2_view4 = np.random.multivariate_normal(clus2_mean_view4, clus2_cov_view4, N)
    clus3_view4 = np.random.multivariate_normal(clus3_mean_view4, clus3_cov_view4, N)
    view4 = np.append(clus1_view4, clus2_view4, axis=0)
    view4 = np.append(view4, clus3_view4, axis=0)    

    permutation = np.random.permutation(y1.shape[0])
    y2 = y1[permutation]
    view3 = view3[permutation]
    view4 = view4[permutation]    

    X1 = np.append(view1, alpha*view3, axis=1)
    X2 = np.append(view4, beta*view2, axis=1)    
    y = y1

    return X1, X2, y    