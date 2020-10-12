# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt


def generate_random_data():
    
    np.random.seed(0)
    x = np.random.rand(100 , 1)
    y = 2 + 3*x + np.random.rand(100 , 1)
    
    return x , y

# plt.scatter(x , y , s = 10)
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.show()
##equation of line , y_pred = mx + b


ms = []
class LinearRegressionUsingGD:
    
    def __init__(self , lr = 0.02 , iterations = 1000):
        self.lr = lr
        self.iterations = iterations
        self.weights = None
        self.bias = 0
        
    def mse(self, y_true , y_pred):
        return np.mean((y_true - y_pred)**2)
        
    def fit(self , x , y):
        
        n_samples , n_features = x.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.iterations):
            
            y_predicted = np.dot(x , self.weights) + self.bias
            dw = (1/n_samples)*np.dot(x.T , (y_predicted - y))
            db = (1/n_samples)*np.sum(y_predicted - y)
            ms.append(self.mse(y , y_predicted))
            self.weights -= self.lr*dw
            self.bias -= self.lr*db
    
    def predict(self , x):
        
        y_predicted = np.dot(x , self.weights) + self.bias
        return y_predicted
    
        
    

        
        
        
        