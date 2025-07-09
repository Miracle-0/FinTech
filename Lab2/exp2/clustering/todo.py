import numpy as np
from scipy.spatial.distance import cdist
from numpy.linalg import inv, eig

def kmeans(X, k):
    '''
    K-Means clustering algorithm

    Input:  x: data point features, N-by-P maxtirx
            k: the number of clusters

    Output:  idx: cluster label, N-by-1 vector
    '''

    N, P = X.shape # N 表示数据点的数量（行数）；P 表示每个数据点的特征数量（列数）
    idx = np.zeros(N) # 初始化 idx 为全零向量，表示每个数据点的初始簇标签
    
    # 选择中心
    centers = X[np.random.choice(N, k, replace=False)]  # 随机选择 k 个数据点作为初始中心
    # 迭代：把每个样本分配到最近的中心，再重新找中心
    for _ in range(100):
        # 计算每个数据点到每个中心的距离，并将每个数据点分配到最近的中心
        dist = cdist(X, centers, 'euclidean')  # 计算每个数据点到每个中心的欧几里得距离
        idx = np.argmin(dist, axis=1)  # 将每个数据点分配到最近的中心
        print(idx)
        
        # 重新计算中心
        new_centers = np.array([X[idx == i].mean(axis=0) for i in range(k)])
        # 如果中心没有变化，则退出迭代
        if np.all(centers == new_centers):
            break
        centers = new_centers 

    return idx

def spectral(W, k):
    '''
    Spectral clustering algorithm

    Input:  W: Adjacency matrix, N-by-N matrix
            k: number of clusters

    Output:  idx: data point cluster labels, N-by-1 vector
    '''
    N = W.shape[0]

    # 计算度矩阵D
    D = np.diag(np.sum(W, axis=1))

    # 构造归一化拉普拉斯矩阵 D^{-1}L
    D_inv = inv(D)
    L_rw = D_inv @ (D - W)

    # 计算特征值和特征向量
    eigenvalues, eigenvectors = eig(L_rw)

    # 取最小的k个特征值对应的特征向量
    idx = np.argsort(np.real(eigenvalues))[:k]

    # 构建K维特征表示
    X = np.real(eigenvectors[:, idx])
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    X = X.astype(float)

    # 进行K-means聚类
    idx = kmeans(X, k)
    return idx

def knn_graph(X, k, threshold):
    '''
    Construct W using KNN graph

    Input:  X:data point features, N-by-P maxtirx.
            k: number of nearest neighbour.
            threshold: distance threshold.

    Output:  W - adjacency matrix, N-by-N matrix.
    '''
    N = X.shape[0]
    W = np.zeros((N, N))
    aj = cdist(X, X, 'euclidean')
    for i in range(N):
        index = np.argsort(aj[i])[:(k+1)]
        W[i, index] = 1
        W[i, i] = 0  # aj[i,i] = 0
    W[aj >= threshold] = 0
    return W
