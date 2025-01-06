import ydf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from scipy.stats import randint
from sklearn.metrics import confusion_matrix
import ipaddress
import os
import matplotlib.pyplot as plt

from sklearn.tree import export_graphviz
from IPython.display import display
import graphviz

#deleted the strings (IP & MAC in this dataset to get it to run)
ds = pd.read_csv("ML Test/output_left_singlelabel_add_del.csv")
print("total dataframe :",ds.shape)

data = ds.values
#ds['sAddress'] = ipaddress.ip_address(ds['sAddress'])
X, y = data[:,:-1], data[:, -1]
print(X.shape, y.shape)
#https://stackoverflow.com/questions/59023756/convert-ip-address-to-integer-in-pandas

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

X_train_ds = pd.DataFrame(X_train)

#train_ds = pd.read_csv("train_ds.csv")
#test_ds = pd.read_csv("test_ds.csv")
#X=ds['sAddress','rAddress']
#y= ds['IT_B_Label']

#X_train, X_test, y_train,  y_test = train_test_split(X,y ,  
                          #random_state=104, 
                          #train_size=0.8, shuffle=True)


random_forest = RandomForestClassifier(n_estimators=25, random_state=42)
random_forest.fit(X_train, y_train)



#https://www.datacamp.com/tutorial/random-forests-classifier-python
#https://www.geeksforgeeks.org/how-to-split-the-dataset-with-scikit-learns-train_test_split-function/
## Split the data into features (X) and target (y) need to do this here.

#random_forest.fit(X_train_ds, y_train_ds)
y_pred = random_forest.predict(X_test)

print("+++++++++++++++++ RF Model +++++++++++++++++++")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

for i in range(3):
    tree = random_forest.estimators_[i]
    dot_data = export_graphviz(tree,
                               feature_names=X_train_ds.columns,  
                               filled=True,  
                               max_depth=2, 
                               impurity=False, 
                               proportion=True)
graph = graphviz.Source(dot_data)
    #name=str(i)+'RF.dot'
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin/'
graph.render(filename='RF.dot')

cm = confusion_matrix(y_test, y_pred)

ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.show()
#plt.savefig('confusion_matrix.jpg')