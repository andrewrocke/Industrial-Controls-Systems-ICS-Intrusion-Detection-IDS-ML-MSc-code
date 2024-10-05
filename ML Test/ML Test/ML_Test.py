print("hello")
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import mglearn example from for book

from sklearn.model_selection import train_test_split
from sklearn import datasets
iris_dataset = datasets.load_iris()
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
X_train, X_test, y_train, y_test = train_test_split(
iris_dataset['data'], iris_dataset['target'], random_state=0)
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))

