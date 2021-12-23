# import pandas as pd
# import xgboost as xgb

# dados = pd.read_csv('C:/Users/Gustavo/Desktop/AdaptiveXGBoostClassifier/datasets/pima-indians-diabetes.data.csv')

# # d_mini_batch_train = xgb.DMatrix(dados, y.astype(int))

# dados.head()

# dados2 = dados.dropna()

# print(dados.shape)
# print(dados2.shape)
########################
# # First XGBoost model for Pima Indians dataset
# from numpy import loadtxt
# from xgboost import XGBClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# # load data
# dataset = loadtxt('./datasets/pima-indians-diabetes.data.csv', delimiter=",")
# # split data into X and y
# X = dataset[:,0:8]
# Y = dataset[:,8]
# # split data into train and test sets
# seed = 10
# test_size = 0.33
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
# # fit model no training data
# model = XGBClassifier(use_label_encoder=False)
# model.fit(X_train, y_train)
# # make predictions for test data
# y_pred = model.predict(X_test)
# predictions = [round(value) for value in y_pred]
# print(y_pred, predictions)
# # evaluate predictions
# accuracy = accuracy_score(y_test, predictions)
# print("Accuracy: %.2f%%" % (accuracy * 100.0))

from sklearn.neighbors import KNeighborsClassifier
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'
## plot decision tree
from numpy import loadtxt
import xgboost as xgb
from xgboost import plot_tree
import matplotlib.pyplot as plt
# load data
dataset = loadtxt('./datasets/pima-indians-diabetes.data.csv', delimiter=",")
# split data into X and y
X = dataset[:,0:8]
y = dataset[:,8]
# fit model no training data


neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, y)
print(neigh.predict([[1,164,82,43,67,32.8,0.341,50]]))
print(neigh.predict_proba([[1,164,82,43,67,32.8,0.341,50]]))