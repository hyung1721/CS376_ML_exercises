from cProfile import label
from turtle import color, distance
import numpy as np


import numpy as np
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
from sklearn import svm

np.random.seed(123123)
n = 100
X1 = np.random.normal(loc=(5, 5), scale=3, size=(n, 2))
X2 = np.random.normal(loc=(20, 20), scale=3, size=(n, 2))
T1 = np.ones(n)
T2 = np.ones(n) * -1

X_train = MinMaxScaler().fit_transform(np.concatenate((X1, X2)))
T_train = np.concatenate((T1, T2))

plt.scatter(X_train[T_train==1][:, 0].T, X_train[T_train==1][:, 1].T, color='y', edgecolors='k', label='label: 1', s = 45)
plt.scatter(X_train[T_train==-1][:, 0].T, X_train[T_train==-1][:, 1].T, color='g', edgecolors='k', label='label: -1', s = 45)
plt.grid(True)
plt.legend()
plt.show()

def compute_loss(C, w, b, X, Y):
    distances = 1 - Y * (X @ w + b)
    distances[distances < 0] = 0.0
    loss = 1 / 2 * np.dot(w, w) + C * np.sum(distances)
    return loss

def compute_w_gradient(C, w, b, X ,Y):
    distances = np.sign(1 - Y * (X @ w + b))
    distances[distances < 0] = 0.0
    dw = C * np.sum((X.T * Y * distances).T, axis=0)
    return w - dw

def compute_b_gradient(C, w, b, X ,Y):
    distances = np.sign(1 - Y * (X @ w + b))
    distances[distances < 0] = 0.0
    db = C * np.sum(Y * distances, axis=0)
    return -db

def predict(w, b, X):
    return np.sign(X @ w + b)

def compute_accuracy(X, Y):
    return np.sum(X == Y) / X.shape[0]

max_epochs = 20
learning_rate = 0.01
C = 1

w = np.random.normal(size=X_train.shape[1])
b = np.concatenate([np.random.normal(size=1)] * 200)

for i in range(max_epochs):
    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)

    X_train_shuffle = X_train[indices]
    T_train_shuffle = T_train[indices]

    w_gradient = compute_w_gradient(C, w, b, X_train_shuffle, T_train_shuffle)
    b_gradient = compute_b_gradient(C, w, b, X_train_shuffle, T_train_shuffle)

    w -= learning_rate * w_gradient
    b -= learning_rate * b_gradient

    loss = compute_loss(C, w, b, X_train_shuffle, T_train_shuffle)
    Y_train = predict(w, b, X_train_shuffle)
    accuracy = compute_accuracy(Y_train, T_train_shuffle)

    print("Epoch {}: Train Loss - {}, Train Accuracy - {}".format(i+1, loss, accuracy))

