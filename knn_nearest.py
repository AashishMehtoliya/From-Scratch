import numpy as np
from collections import Counter


def euclidean_distance(x1 , x2):
    return np.sqrt(np.sum((x1 - x2)**2))
    
    
class kNN():

    def __init__(self , k=3):
        self.k = k

    def fit(self , X , y):
        self.X_train = X
        self.y_train = y
        
        
    def predict(self , X):
        
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)
    
    def _predict(self , x):
        
        distances = [euclidean_distance(x , x_train) for x_train in self.X_train] 
        
        k_indicies = np.argsort(distances)[:self.k]
        k_nearest_ix = [self.y_train[i] for i in k_indicies]
        
        most_common_class = Counter(k_nearest_ix).most_common(1)[0][0]
        
        return most_common_class
        
        
        
        
        
        
        
        
        