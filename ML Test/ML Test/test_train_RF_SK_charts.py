import ydf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, roc_curve,  f1_score, roc_auc_score, auc
from scipy.stats import randint
#from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import ipaddress
import os
import matplotlib.pyplot as plt

from sklearn.tree import export_graphviz
from IPython.display import display
import graphviz
import joblib

#deleted the strings (IP & MAC in this dataset to get it to run)
ds = pd.read_csv("ML Test/output_left_singlelabel_add_del_NST.csv")
print("total dataframe :",ds.shape)

data = ds.values
#ds['sAddress'] = ipaddress.ip_address(ds['sAddress'])
X, y = data[:,:-1], data[:, -1]
print("x data, y data", X.shape, y.shape)
#https://stackoverflow.com/questions/59023756/convert-ip-address-to-integer-in-pandas

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)
print("xtrain", X_train.shape, "xtest",X_test.shape,"ytrain", y_train.shape, "ytest",y_test.shape)

X_train_ds = pd.DataFrame(X_train)
X_test_ds = pd.DataFrame(X_test)
y_train_ds = pd.DataFrame(y_train)
y_test_ds = pd.DataFrame(y_test)
print("xtrain", X_train_ds.head())
print("xtest", X_test_ds.head())
print("ytrain", y_train_ds.head())
print("ytest", y_test_ds.head())
#train_ds = pd.read_csv("train_ds.csv")
#test_ds = pd.read_csv("test_ds.csv")
#X=ds['sAddress','rAddress']
#y= ds['IT_B_Label']

#X_train, X_test, y_train,  y_test = train_test_split(X,y ,  
                          #random_state=104, 
                          #train_size=0.8, shuffle=True)


model = RandomForestClassifier(n_estimators=25, random_state=42, max_depth=200)
model.fit(X_train, y_train)



#https://www.datacamp.com/tutorial/random-forests-classifier-python
#https://www.geeksforgeeks.org/how-to-split-the-dataset-with-scikit-learns-train_test_split-function/
## Split the data into features (X) and target (y) need to do this here.

#random_forest.fit(X_train_ds, y_train_ds)
y_pred = model.predict(X_test)

print("+++++++++++++++++ RF Model +++++++++++++++++++")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

for i in range(3):
    tree = model.estimators_[i]
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

print("ROC")
#https://www.geeksforgeeks.org/how-to-plot-roc-curve-in-python/
#fpr, tpr, thresholds = roc_curve(y_test, y_pred)
y_proba = model.predict_proba(X_test)[:,1]
roc_auc = roc_auc_score(y_test, y_proba)
f1 = f1_score(y_test, y_pred)
print (f"AUC - ROC Score: {roc_auc:.2f}")
print(f"F1 Score: {f1:.2f}")
#plt.figure()  
#plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
#plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
#plt.xlim([0.0, 1.0])
#plt.ylim([0.0, 1.05])
#plt.xlabel('False Positive Rate')
#plt.ylabel('True Positive Rate')
#plt.title('ROC Curve for Breast Cancer Classification')
#plt.legend()
#plt.show()
#plt.savefig("roc.jpg")

#https://www.datacamp.com/tutorial/auc
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

#fig, axs = plt.subplots(2)

#fig.suptitle('Metrics')
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")

print("CM")
cm = confusion_matrix(y_test, y_pred)

ConfusionMatrixDisplay(confusion_matrix=cm).plot()
#axs[1].plot.savefig("cm.jpg")
plt.savefig('plots.jpg')

plt.show()

#https://mljar.com/blog/save-load-random-forest/
# save
joblib.dump(model, "my_RF_SK_model.joblib")

