# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 00:55:32 2020

@author: Aashish Mehtoliya
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

# import LinearRegression

x , y = datasets.make_regression(n_samples = 100 , n_features = 1 , noise = 20 , random_state =1)
X_train , X_test , y_train , y_test = train_test_split(x , y, test_size = 0.2 , random_state = 1234)

import Linear_Regression
from Linear_Regression import LinearRegressionUsingGD

regressor = LinearRegressionUsingGD(lr = 0.01 , iterations = 2000)
regressor.fit(X_train , y_train)
predicted = regressor.predict(X_test)

def mse(y_true , y_pred):
    return np.mean((y_true - y_pred)**2)

mse_value = mse(y_test , predicted)

print(mse_value)
y_predicted_line = regressor.predict(x)

print(Linear_Regression.ms)

cmap = plt.get_cmap('viridis')
fig = plt.figure(figsize=(8 , 6))
m1 = plt.scatter(X_train , y_train , color = cmap(0.9) , s = 10)
m2 = plt.scatter(X_test , y_test , color = cmap(0.5) , s = 10)
plt.plot(x , y_predicted_line , color = 'red' , linewidth=2 , label = 'pred')
plt.show()










