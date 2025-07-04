# 导入库和数据
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

train_rating = pd.read_csv("ratings_train_v1.csv") 
test_rating = pd.read_csv("ratings_test_v1.csv") 

all_data = pd.concat([train_rating, test_rating])

all_user = np.unique(all_data['userId'])  # 用户数量
all_item = np.unique(all_data['movieId'])  # 物品数量

# 创建对应数量的空评分表格
num_user = len(all_user)
num_item = len(all_item)

rating_mat = np.zeros([num_user, num_item], dtype=int)

for i in range(len(train_rating)):
    user = int(train_rating.iloc[i]['userId'])  # 用户编号
    item = int(train_rating.iloc[i]['movieId'])  # 物品编号
    score = train_rating.iloc[i]['rating']  # 评分

    user_id = np.where(all_user == user)[0][0]  # 获取用户在 all_user 中的索引
    item_id = np.where(all_item == item)[0][0]  # 获取物品在 all_item 中的索引

    rating_mat[user_id, item_id] = float(score)  # 构建评分矩阵


def get_loss(rating_mat, embed_dim, lamda, P, Q):
    m, n = rating_mat.shape
    error = 0
    for i in range(m):
        for j in range(n):
            if rating_mat[i, j] > 0:
                eij = rating_mat[i, j] - np.dot(P[i, :], Q[j, :].T)
                error += eij ** 2
    error += lamda * (np.sum(P ** 2) + np.sum(Q ** 2))  # 正则化项
    return error


def matrix_factorization(rating_mat, embed_dim, gamma, lamda, steps):
    m, n = rating_mat.shape
    P = np.random.rand(m, embed_dim)
    Q = np.random.rand(n, embed_dim)
    error_list = []

    for step in range(steps):
        for i in range(m):
            for j in range(n):
                if rating_mat[i, j] > 0:
                    eij = rating_mat[i, j] - np.dot(P[i, :], Q[j, :].T)

                    # 更新规则
                    P[i, :] += gamma * (eij * Q[j, :] - lamda * P[i, :])
                    Q[j, :] += gamma * (eij * P[i, :] - lamda * Q[j, :])
        error = get_loss(rating_mat, embed_dim, lamda, P, Q)
        error_list.append(error)

    return P, Q, error_list


def non_negative_matrix_factorization(rating_mat, embed_dim, gamma, lamda, steps):
    m, n = rating_mat.shape
    P = np.random.rand(m, embed_dim)
    Q = np.random.rand(n, embed_dim)
    error_list = []

    for step in range(steps):
        for i in range(m):
            for j in range(n):
                if rating_mat[i, j] > 0:
                    eij = rating_mat[i, j] - np.dot(P[i, :], Q[j, :].T)

                    # 更新规则
                    P[i, :] += gamma * (eij * Q[j, :] - lamda * P[i, :])
                    Q[j, :] += gamma * (eij * P[i, :] - lamda * Q[j, :])

                    # 非负约束
                    P[i, :] = np.clip(P[i, :], 0, None)
                    Q[j, :] = np.clip(Q[j, :], 0, None)
        error = get_loss(rating_mat, embed_dim, lamda, P, Q)
        error_list.append(error)

    return P, Q, error_list


def test(test_rating, P, Q, all_user, all_item):
    error = 0
    count = 0

    for _, row in test_rating.iterrows():
        user = int(row['userId'])
        item = int(row['movieId'])
        true_rating = row['rating']

        user_id = np.where(all_user == user)[0][0]
        item_id = np.where(all_item == item)[0][0]

        pred_rating = np.dot(P[user_id, :], Q[item_id, :].T)
        eij = true_rating - pred_rating

        error += eij ** 2
        count += 1

    mse = error / count
    rmse = np.sqrt(mse)
    return mse, rmse


def draw(error_list):
    plt.plot(range(len(error_list)), error_list)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training Loss Curve")
    plt.show()


def train(rating_mat, all_user, all_item, embed_dim=32, gamma=0.001, lamda=0.01, steps=20, use_nmf=False):
    if use_nmf:
        P, Q, error_list = non_negative_matrix_factorization(rating_mat, embed_dim, gamma, lamda, steps)
    else:
        P, Q, error_list = matrix_factorization(rating_mat, embed_dim, gamma, lamda, steps)

    mse, rmse = test(test_rating, P, Q, all_user, all_item)
    return mse, rmse, error_list


def run_hyperparameter_search():
    results = []

    # 设置要测试的参数组合
    embed_dims = [32]
    gammas = [0.05]
    lambdas = [0.5]
    step_list = [5, 10, 20, 30, 50, 100]
    use_nmfs = [True, False]

    for embed_dim, gamma, lamda, steps, use_nmf in itertools.product(
        embed_dims, gammas, lambdas, step_list, use_nmfs):

        print(f"Testing: embed_dim={embed_dim}, gamma={gamma}, lamda={lamda}, steps={steps}, NMF={use_nmf}")

        try:
            mse, rmse, _ = train(rating_mat, all_user, all_item, embed_dim, gamma, lamda, steps, use_nmf)
            results.append({
                'embed_dim': embed_dim,
                'gamma': gamma,
                'lamda': lamda,
                'steps': steps,
                'use_nmf': use_nmf,
                'mse': mse,
                'rmse': rmse
            })
        except Exception as e:
            print(f"Failed with params {embed_dim}, {gamma}, {lamda}, {steps}, {use_nmf}: {e}")

    return pd.DataFrame(results)


# 开始超参数搜索并保存结果
print("开始进行超参数搜索...")
results_df = run_hyperparameter_search()
results_df.to_csv("new_hyperparameter_search_results.csv", index=False)
print("结果已保存至 new_hyperparameter_search_results.csv")

