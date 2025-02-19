import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_decision_forests as tfdf
from urllib.parse import unquote_to_bytes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt # Import matplotlib


ds = pd.read_csv("/output_left_singlelabel_add_del_NST.csv")
print("total dataframe :",ds.shape)
#print(list(ds))
#print(ds.head())
#X=ds['sAddress','rAddress']
#y= ds['IT_B_Label']
mask = np.random.rand(len(ds)) < 0.8
train_ds = ds[mask]
test_ds = ds[~mask]
#print("train_ds",train_ds.head())
#print("test_ds",test_ds.head())

tf_train_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds, label="NST_B_Label")
tf_serving_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(test_ds, label="NST_B_Label")

#for features, label in tf_train_dataset:
  #print("Features:",features)
  #print("label:", label)
#data = ds.values
model = tfdf.keras.RandomForestModel(verbose=2)
model.fit(tf_train_dataset)
model.summary()

model.compile(metrics=["accuracy"])
evaluation = model.evaluate(tf_serving_dataset, return_dict=True)
print()

#for name, value in evaluation.items():
#  print(f"{name}: {value:.4f}")

# Predict on the test set
y_pred_probs = model.predict(tf_serving_dataset)
y_pred = np.argmax(y_pred_probs, axis=1) # Get predicted class labels

# Extract true labels from the tf_serving_dataset
y_true = np.concatenate([y for x, y in tf_serving_dataset], axis=0)

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Display confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(np.unique(y_true))) # Assuming y_true has all possible class labels
plt.xticks(tick_marks, np.unique(y_true), rotation=45)
plt.yticks(tick_marks, np.unique(y_true))

plt.ylabel('True label')
plt.xlabel('Predicted label')

# Add text annotations to the confusion matrix cells
thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.show()

