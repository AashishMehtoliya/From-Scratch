# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 17:32:13 2020

@author: Aashish Mehtoliya
"""


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

from logistic_regression import LogisticRegression

bc = datasets.load_breast_cancer()
# bc_dt = pd.DataFrame(bc)
x , y = bc.data , bc.target
# print(type(x.feature_names))

X_train, X_test, y_train, y_test = train_test_split(x , y , test_size = 0.2 , random_state = 4)

# X_train = X_train/max(X_train)
# X_test = X_test/max(X_test)
# y_train = y_train/max(y_train)
# y_test = y_test/max(y_test)

def accuracy(y_true , y_predicted):
    accuracy = np.sum(y_true==y_predicted)/len(y_true)
    return accuracy

regressor = LogisticRegression()
regressor.fit(X_train , y_train)

y_pred = regressor.predict(X_test)

# print()

# sigmoid = lambda x: 1 / (1 + np.exp(-x))
# x=np.linspace(-10,10,100)
# fig = plt.figure()
# m1 = plt.scatter(X_train , y_train , s = 10 , color = 'red')
# m2 = plt.scatter(X_test , y_test , s = 10 , color = 'yellow')
# plt.plot(x,sigmoid(x),'b', label='linspace(-10,10,100)')
# plt.show()
print(accuracy(y_test , y_pred))

    