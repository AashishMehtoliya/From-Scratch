import pandas as pd
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
from knn_nearest import kNN

cmap = ListedColormap(["#FF0000" , "#00FF00" , "#0000FF"])

iris = datasets.load_iris()
x , y = iris.data , iris.target

X_train , X_test , y_train , y_test = train_test_split(x , y , test_size = 0.2 , random_state  = 1234)

# plt.figure()
# plt.scatter(x[:,1], y ,c = y, cmap = cmap , edgecolors='k' , s = 20)
# plt.show()

clf = kNN(k = 5)

clf.fit(X_train , y_train)

predicted_class = clf.predict(X_test)

acc = np.sum(predicted_class == y_test)/len(y_test)

print(acc)



