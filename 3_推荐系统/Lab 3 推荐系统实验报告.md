# Lab 3 推荐系统

## 实验目的

通过矩阵分解，实现推荐系统



## 实验流程

### 先导入各种数据集

这一部分代码已给出，不赘述

### 将得到的csv文件转为对应的rating矩阵

此处代码文档也已给出，不赘述

### 选择矩阵分解模型

选用了两个隐向量矩阵，并不断进行优化

### 定义损失函数与正则化，以及非负矩阵分解

本次实验使用的是带有正则化的平方误差损失函数

```Python
error = error + lamda * (np.sum(P ** 2) + np.sum(Q ** 2)) # 添加正则化项
```

### 优化求解（梯度下降）

```Python
eij = rating_mat[i, j] - np.dot(P[i, :], Q[j, :].T) #作差

# 梯度下降
P[i, :] += gamma * (eij * Q[j, :] - lamda * P[i, :])
Q[j, :] += gamma * (eij * P[i, :] - lamda * Q[j, :])
```

### 生成推荐

根据测试集，判断效果

经典情况：embed_dim = 32, gamma = 0.001, lamda = 0.01, steps = 20

Test MSE:  17.340639399126598 
Test RMSE:  4.164209336612005

<img src="/Users/andy/Desktop/exp3/Figure_1.png" alt="Figure_1" style="zoom: 50%;" />

### 评估和调参

修改embed_dim（隐向量矩阵的维度）, gamma（学习率）, lamda（正则化参数）, steps（迭代次数）

进而通过调参，找到最好的结果



## 实验结果

### 隐向量矩阵维度embed_dim：

保持gamma=0.01，lamda=0.1，steps=20，使用非负矩阵分解

结果显示，矩阵维度过高，容易造成过拟合，从而使偏差变大

<img src="/Users/andy/Library/Application Support/typora-user-images/image-20250702203046367.png" alt="image-20250702203046367" style="zoom:50%;" />

​	附：随着隐向量矩阵维度增加，训练集拟合情况。其实拟合程度没有像想象中提升得那么大

<img src="/Users/andy/Library/Application Support/typora-user-images/image-20250702204414227.png" alt="image-20250702204414227" style="zoom:50%;" />

### gamma（学习率）

如下表所示，控制其他参数相同，学习率增加，训练效果变好

值得注意的是，当学习率过高，如设为0.1、0.5时，会使更新幅度过大，进而使下一轮误差更大，最终陷入恶性循环，无法收敛

| embed_dim | gamma | lamda | steps | use_nmf | mse   | rmse |
| --------- | ----- | ----- | ----- | ------- | ----- | ---- |
| 32        | 0.001 | 0.1   | 20    | TRUE    | 18.76 | 4.33 |
| 32        | 0.01  | 0.1   | 20    | TRUE    | 13.75 | 3.71 |
| 32        | 0.05  | 0.1   | 20    | TRUE    | 8.88  | 2.98 |

<img src="/Users/andy/Library/Application Support/typora-user-images/image-20250704124250129.png" alt="image-20250704124250129" style="zoom:50%;" />

### lamda（正则化参数）

正则化系数主要用于减少每一步的变化量，进而避免过拟合，训练得到更简单的模型，即让参数较小，避免某一个特征对模型产生过大影响（比如噪声）

同时也有利于损失函数收敛

| embed_dim | gamma | lamda | steps | use_nmf | mse   | rmse |
| --------- | ----- | ----- | ----- | ------- | ----- | ---- |
| 32        | 0.05  | 0     | 20    | TRUE    |       |      |
| 32        | 0.05  | 0.01  | 20    | TRUE    |       |      |
| 32        | 0.05  | 0.03  | 20    | TRUE    | 18.90 | 4.35 |
| 32        | 0.05  | 0.05  | 20    | TRUE    | 14.91 | 3.86 |
| 32        | 0.05  | 0.1   | 20    | TRUE    | 9.20  | 3.03 |
| 32        | 0.05  | 0.5   | 20    | TRUE    | 7.58  | 2.75 |
| 32        | 0.05  | 1     | 20    | TRUE    | 5.73  | 2.39 |

<img src="/Users/andy/Library/Application Support/typora-user-images/image-20250704125209102.png" alt="image-20250704125209102" style="zoom:50%;" />

### steps（迭代次数）

如下表所示，随着迭代次数增加，训练效果有所改善

| embed_dim | gamma | lamda | steps | use_nmf | mse  | rmse |
| --------- | ----- | ----- | ----- | ------- | ---- | ---- |
| 32        | 0.05  | 0.5   | 5     | TRUE    | 8.14 | 2.85 |
| 32        | 0.05  | 0.5   | 10    | TRUE    | 7.32 | 2.70 |
| 32        | 0.05  | 0.5   | 20    | TRUE    | 6.67 | 2.58 |
| 32        | 0.05  | 0.5   | 30    | TRUE    | 7.40 | 2.72 |
| 32        | 0.05  | 0.5   | 50    | TRUE    | 5.74 | 2.40 |
| 32        | 0.05  | 0.5   | 100   | TRUE    | 6.12 | 2.47 |

<img src="/Users/andy/Library/Application Support/typora-user-images/image-20250704131416751.png" alt="image-20250704131416751" style="zoom:50%;" />

### 是否采用非负矩阵分解NFM

感觉采用了以后结果好一点，但相差不大

下表结果为在不同迭代次数下，是否采用NFM对结果的影响

| embed_dim | gamma | lamda | steps | use_nmf | mse  | rmse |
| --------- | ----- | ----- | ----- | ------- | ---- | ---- |
| 32        | 0.05  | 0.5   | 5     | FALSE   | 8.72 | 2.95 |
| 32        | 0.05  | 0.5   | 5     | TRUE    | 8.34 | 2.89 |
| 32        | 0.05  | 0.5   | 10    | FALSE   | 7.18 | 2.68 |
| 32        | 0.05  | 0.5   | 10    | TRUE    | 7.57 | 2.75 |
| 32        | 0.05  | 0.5   | 20    | FALSE   | 6.90 | 2.63 |
| 32        | 0.05  | 0.5   | 20    | TRUE    | 7.07 | 2.66 |
| 32        | 0.05  | 0.5   | 30    | FALSE   | 7.89 | 2.81 |
| 32        | 0.05  | 0.5   | 30    | TRUE    | 7.84 | 2.80 |
| 32        | 0.05  | 0.5   | 50    | FALSE   | 6.72 | 2.59 |
| 32        | 0.05  | 0.5   | 50    | TRUE    | 6.78 | 2.60 |
| 32        | 0.05  | 0.5   | 100   | FALSE   | 6.41 | 2.53 |
| 32        | 0.05  | 0.5   | 100   | TRUE    | 6.18 | 2.49 |

<img src="/Users/andy/Library/Application Support/typora-user-images/image-20250704133529603.png" alt="image-20250704133529603" style="zoom:50%;" />





## 附件：源代码

```python
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
    
    ######################
    #Please code for task 5 (non_negative_matrix_factorization) here#

    ######################
    
    # return P, Q, error_list


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







```

