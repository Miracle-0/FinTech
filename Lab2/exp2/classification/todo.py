import numpy as np

# You may find below useful for Support Vector Machine
# More details in
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
# from scipy.optimize import minimize

def func(X, y):
    '''
    Classification algorithm.

    Input:  X: Training sample features, P-by-N
            y: Training sample labels, 1-by-N

    Output: w: learned parameters, (P+1)-by-1
    '''
    P, N = X.shape
    w = np.zeros((P + 1, 1))  # 初始化权重参数（包括偏置项）

    # 将偏置项加入数据集
    X_bias = np.vstack((np.ones((1, N)), X))

    # 感知机训练
    for i in range(100):  # 迭代次数
        mis_classified = False
        for j in range(N):
            if y[0, j] * np.dot(w.T, X_bias[:, j:j+1]) <= 0:
                # 更新权重
                w += y[0, j] * X_bias[:, j:j+1]
                mis_classified = True
        if not mis_classified:
            break

    return w