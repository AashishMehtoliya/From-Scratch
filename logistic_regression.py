# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 04:05:50 2020

@author: Aashish Mehtoliya
"""


import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression:
    
    def __init__(self , lr = 0.01 , iterations = 1000):
        self.lr = lr
        self.iterations = iterations
        self.weights = None
        self.bias = None
    
    def fit(self , x, y):
        
        n_samples , n_features = x.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.iterations):
            
            linear_model = np.dot(x , self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)
        
            dw = (1/n_samples)*np.dot(x.T , (y_predicted - y))
            db = (1/n_samples)*np.sum(y_predicted - y)           
            
            self.weights -= self.lr*dw
            self.bias -= self.lr*db
            
            
    def predict(self , x):
        linear_model = np.dot(x , self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        
        y_predicted_cls = [1 if i >= 0.5 else 0 for i in y_predicted]
        
        return y_predicted_cls
        
    def _sigmoid(self , x):
        return 1/(1+np.exp(-x))
        