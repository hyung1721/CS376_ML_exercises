from contextlib import redirect_stderr
from logging import basicConfig
from mailbox import NoSuchMailboxError
import re
from tkinter.tix import Y_REGION
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.multioutput import RegressorChain

class RidgeRegression():
    def __init__(self, regularization_factor):
        self.regularization_factor = regularization_factor
    
    def fit(self, X, Y, epochs, learning_rate, batch_size):
        num_examples, num_features = X.shape

        self.W = np.random.rand(num_features)
        self.b = 0
        self.train_loss = []
        
        num_iterations_per_epochs = int(num_examples/batch_size)

        for ep in range(epochs):
            if num_iterations_per_epochs != 1:
                df_tmp = pd.concat([pd.DataFrame(X), pd.DataFrame(Y)], axis=1).sample(frac=1)
                X = df_tmp.iloc[:, :-1].values
                Y = df_tmp.iloc[:, -1].values
            
            for i in range(num_iterations_per_epochs):
                X_batch = X[batch_size*i : batch_size*(i+1)]
                Y_batch = Y[batch_size*i : batch_size*(i+1)]

                dW, db = self.calculate_gradient(X_batch, Y_batch)

                self.W = self.W - learning_rate * dW
                self.b = self.b = learning_rate * db
            
            loss = self.calculate_loss(X, Y)
            self.train_loss.append(loss)
        
        return self
    
    def predict(self, X):
        return X.dot(self.W) + self.b
    
    def calculate_gradient(self, X, Y):
        num_examples = X.shape[0]
        Y_pred = self.predict(X)
        regularization_term = 2 * (self.regularization_factor * self.W)
        
        dW = - ((X.T).dot(Y-Y_pred)) / num_examples + regularization_term
        db = - np.sum(Y-Y_pred) / num_examples

        return dW, db
    
    def calculate_loss(self, X, Y):
        num_examples = X.shape[0]
        Y_pred = self.predict(X)
        regularization_term = self.regularization_factor * np.sum(np.square(self.W))

        loss = np.sum(np.square(Y-Y_pred)) / (2 * num_examples) + regularization_term

        return loss
    
    def eval(self, X, Y):
        num_examples = X.shape[0]
        Y_pred = self.predict(X)

        RMSE = np.sqrt(np.sum(np.square(Y_pred - Y)) / num_examples)

        return RMSE

dataset_path = './data_for_regression_py.csv'
df = pd.read_csv(dataset_path, sep='|')
print(df.head(5))

X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=35)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

regularization_factor = 0.1
ridge_model = RidgeRegression(regularization_factor)

epochs = 2000
learning_rate = 0.01
batch_size = X_train.shape[0]
ridge_model.fit(X_train, Y_train, epochs, learning_rate, batch_size)

loss = ridge_model.train_loss
x_axis = np.arange(0, len(loss))
plt.plot(x_axis, loss, label='Ridge')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# print("rmse (test): ", ridge_model.eval(X_test, Y_test))

# regularization_factor = 0.1
# ridge_model_full = RidgeRegression(regularization_factor)
# ridge_model_16   = RidgeRegression(regularization_factor)
# ridge_model_1    = RidgeRegression(regularization_factor)

# epochs = 2000
# learning_rate = 0.01

# ridge_model_full = ridge_model

# batch_size = 16
# ridge_model_16.fit(X_train, Y_train, epochs, learning_rate, batch_size)

# batch_size = 1
# ridge_model_1.fit(X_train, Y_train, epochs, learning_rate, batch_size)

# loss_full, loss_16, loss_1 = ridge_model_full.train_loss, ridge_model_16.train_loss, ridge_model_1.train_loss
# x_axis = np.arange(0, len(loss_full))
# plt.plot(x_axis, loss_full, label='Batch size = Full')
# plt.plot(x_axis, loss_16, label='Batch size = 16')
# plt.plot(x_axis, loss_1, label='Batch size = 1')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.ylim(0,1)
# plt.legend()
# plt.show()

# ridge_model_full.eval(X_test, Y_test), ridge_model_16.eval(X_test, Y_test), ridge_model_1.eval(X_test, Y_test)

X = df.iloc[:, :-1].values
# pca_test = PCA(n_components=X.shape[1])
# pca_test.fit(X)
# print("Eigen values: ", pca_test.explained_variance_)
# print("Cumulative explained variances: ")
# for i in range(1, len(pca_test.explained_variance_) + 1):
#     print("# PCs = {}: {}".format(i, round(sum(pca_test.explained_variance_ratio_[:i]), 2)))

pca = PCA(n_components=8)
pca.fit(X)

X2_train = pca.transform(X_train)
X2_test  = pca.transform(X_test)
Y2_train, Y2_test = Y_train, Y_test
X2_train.shape, X2_test.shape, Y2_train.shape, Y2_test.shape

ridge_model_rf01 = RidgeRegression(regularization_factor=0.1)
ridge_model_rf0  = RidgeRegression(regularization_factor=0)

epochs = 2000
learning_rate = 0.01
ridge_model_rf01.fit(X2_train, Y2_train, epochs, learning_rate, batch_size)
ridge_model_rf0.fit(X2_train, Y2_train, epochs, learning_rate, batch_size)

print("Ridge : ", ridge_model_rf01.eval(X2_test, Y2_test))
print("Linear : ", ridge_model_rf0.eval(X2_test, Y2_test))