import numpy as np
from gen_data import gen_data
from plot import plot
from todo import func

no_iter = 1000  # number of iteration
no_train = 70  # 70% 作为训练数据
no_test = 30   # 30% 作为测试数据
no_data = 100  # number of all data
assert(no_train + no_test == no_data)

cumulative_train_err = 0
cumulative_test_err = 0

for i in range(no_iter):
    X, y, w_gt = gen_data(no_data)
    X_train, X_test = X[:, :no_train], X[:, no_train:]
    y_train, y_test = y[:, :no_train], y[:, no_train:]
    w_l = func(X_train, y_train)
    # Compute training and testing error
    X_train_bias = np.vstack((np.ones((1, no_train)), X_train))
    X_test_bias = np.vstack((np.ones((1, no_test)), X_test))

    train_preds = np.sign(np.dot(w_l.T, X_train_bias))
    test_preds = np.sign(np.dot(w_l.T, X_test_bias))

    train_err = np.mean(train_preds[0] != y_train[0])
    test_err = np.mean(test_preds[0] != y_test[0])
    cumulative_train_err += train_err
    cumulative_test_err += test_err

train_err = cumulative_train_err / no_iter
test_err = cumulative_test_err / no_iter

plot(X, y, w_gt, w_l, "Classification")
print("Training error: %s" % train_err)
print("Testing error: %s" % test_err)