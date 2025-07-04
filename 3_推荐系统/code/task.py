# 导入库和数据
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_rating = pd.read_csv("ratings_train_v1.csv") 
test_rating = pd.read_csv("ratings_test_v1.csv") 


all_data = pd.concat([train_rating, test_rating])

all_user = np.unique(all_data['userId']) # 用户数量
all_item = np.unique(all_data['movieId']) # 物品数量

#  创建对应数量的空评分表格
num_user = len(all_user)
num_item = len(all_item)

rating_mat = np.zeros([num_user, num_item], dtype=int)

for i in range(len(train_rating)):
    user = int(train_rating.iloc[i]['userId']) # 用户编号
    item = int(train_rating.iloc[i]['movieId']) # 物品编号
    score = train_rating.iloc[i]['rating'] # 评分
    
    user_id = np.where(all_user == user)[0][0]  #使用 np.where 查找当前 user 和 item 在全局数组 all_user 和 all_item 中的索引位置
    item_id = np.where(all_item == item)[0][0]  

    rating_mat[user_id, item_id] = float(score) #利用这两个索引在评分矩阵 rating_mat 中定位，并将 score 转换为浮点数后赋值给对应位置

def get_loss(rating_mat, embed_dim, lamda, P, Q):
    '''
    INPUT:
    :param rating_mat: [m, n] 用户评分矩阵
    :param embed_dim: dimension of the embeddings 隐向量的维度
    :param gamma: learning rate 学习率
    :param P: user embedding [m, embed_dim] 用户隐向量矩阵
    :param Q: item embedding [n, embed_dim] 物品隐向量矩阵
    RETURN: the sum of the error. 误差值
    '''
    
    m, n = rating_mat.shape
    error = 0
    for i in range(m):
            for j in range(n):
                if rating_mat[i, j] > 0: # 对于已知项进行训练
                    eij = rating_mat[i, j] - np.dot(P[i, :], Q[j, :].T) #作差
                    error += eij ** 2 # 计算误差平方和
    error = error + lamda * (np.sum(P ** 2) + np.sum(Q ** 2)) # 添加正则化项
    return error
    

def matrix_factorization(rating_mat, embed_dim, gamma, lamda, steps):
    '''
    INPUT:
    :param rating_mat: [m, n] 用户评分矩阵
    :param embed_dim: dimension of the embeddings 隐向量的维度
    :param gamma: learning rate 学习率
    :param lamda: balanced hyper parameters 正则化参数
    :param steps: training epoch 迭代次数
    RETURN:
    :param P: user embedding [m, embed_dim] 用户隐向量矩阵
    :param Q: item embedding [embed_dim, n] 物品隐向量矩阵
    :param error_list: list 每次迭代的损失值列表
    '''
    m, n = rating_mat.shape
    P = np.random.rand(m, embed_dim)  # 用户隐向量矩阵
    Q = np.random.rand(n, embed_dim)  # 物品隐向量
    error_list = []
    
    for step in range(steps):
        for i in range(m):
            for j in range(n):
                if rating_mat[i, j] > 0: # 对于已知项进行训练
                    eij = rating_mat[i, j] - np.dot(P[i, :], Q[j, :].T) #作差

                    # 梯度下降
                    P[i, :] += gamma * (eij * Q[j, :] - lamda * P[i, :])
                    Q[j, :] += gamma * (eij * P[i, :] - lamda * Q[j, :])
        # 计算损失
        if step%1==0:
            error = get_loss(rating_mat, embed_dim, lamda, P, Q)
            error_list.append(error)
    
    return P, Q, error_list



def test(test_rating, P, Q, all_user, all_item):
    '''
    INPUT:
    :param test_rating:  
    :param all_user: the all_user lists
    :param all_item: the all_item lists
    :param P: user embedding [m, embed_dim] 用户隐向量矩阵
    :param Q: item embedding [n, embed_dim] 商品隐向量矩阵
    RETURN: the mse and rmse on the test samples.
    '''
    
    error = 0
    count = 0
    
    # 遍历 test_rating 中的每一行真实评分
    for _, row in test_rating.iterrows():
        user = int(row['userId'])
        item = int(row['movieId'])
        true_rating = row['rating']
        
        # 映射到 rating_mat 中的索引
        user_id = np.where(all_user == user)[0][0]
        item_id = np.where(all_item == item)[0][0]
        
        # 预测评分
        pred_rating = np.dot(P[user_id, :], Q[item_id, :].T)
        eij = true_rating - pred_rating
        
        error += eij ** 2
        count += 1

    mse = error / count
    rmse = np.sqrt(mse)
    return mse, rmse


def non_negative_matrix_factorization(rating_mat, embed_dim, gamma, lamda, steps):
    '''
    INPUT:
    :param rating_mat: [m, n]
    :param embed_dim: dimension of the embeddings
    :param gamma: learning rate
    :param lamda: balanced hyper parameters
    :param steps: training epoch
    RETURN:
    :param P: user embedding [m, embed_dim] 
    :param Q: item embedding [embed_dim, n]
    :param error_list: list
    '''
    m, n = rating_mat.shape
    P = np.random.rand(m, embed_dim)
    Q = np.random.rand(n, embed_dim)
    error_list = []

    for step in range(steps):
        for i in range(m):
            for j in range(n):
                if rating_mat[i, j] > 0:
                    eij = rating_mat[i, j] - np.dot(P[i, :], Q[j, :].T)

                    # 梯度下降部分（保持非负）
                    P[i, :] += gamma * (eij * Q[j, :] - lamda * P[i, :])
                    Q[j, :] += gamma * (eij * P[i, :] - lamda * Q[j, :])

                    # 强制非负
                    P[i, :] = np.clip(P[i, :], 0, None)
                    Q[j, :] = np.clip(Q[j, :], 0, None)

        # 每个 epoch 计算一次损失
        error = get_loss(rating_mat, embed_dim, lamda, P, Q)
        error_list.append(error)

    return P, Q, error_list


def draw(error_list):
    plt.plot(range(len(error_list)), error_list)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()

def train(rating_mat, all_user, all_item, embed_dim, gamma, lamda, steps):
    P, Q, error_list = matrix_factorization(rating_mat, embed_dim, gamma, lamda, steps)
    mse, rmse = test(test_rating, P, Q, all_user, all_item)
    
    print('Test MSE: ', mse, 'Test RMSE: ', rmse)
    draw(error_list)

train(rating_mat, all_user, all_item, embed_dim = 32, gamma = 0.001, lamda = 0.01, steps = 20)