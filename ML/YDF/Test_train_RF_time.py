import ydf
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import tensorflow as tf
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
import time

#import tensorflow_decision_forests as tfdf



train_ds = pd.read_csv("train_ds.csv")
test_ds = pd.read_csv("test_ds.csv")
#X=ds['sAddress','rAddress']
#y= ds['IT_B_Label']

#X_train, X_test, y_train,  y_test = train_test_split(X,y ,  
                          #random_state=104, 
                          #train_size=0.8, shuffle=True)

print("train_ds count :" ,len(train_ds))
#test_ds = pd.read_csv("ML Test/output_left.csv")
print("test_ds count :" , len(test_ds))
#train_ds = pd.read_csv("ML Test/output_left_23.csv")
print("train_ds :" ,train_ds.head(5))
#test_ds = pd.read_csv("ML Test/output_left.csv")
print("test_ds :" ,test_ds.head(5))
#model = ydf.GradientBoostedTreesLearner(label="IT_B_Label").train(train_ds)
#model.describe()

learner = ydf.RandomForestLearner(label="NST_B_Label",
                                  compute_oob_performances=True)
start_time = time.time()





model = learner.train(train_ds)
end_time = time.time()
train_time = end_time - start_time

print(f"Train time: {train_time} seconds")

oob_evaluations = model.out_of_bag_evaluations()
print(oob_evaluations[-1].evaluation)
print("+++++++++++++++++ RF Model +++++++++++++++++++")
print(model.evaluate(test_ds))
#print(model.benchmark(test_ds))
#model.analyze(test_ds, sampling=0.1)

# Look at a model (input features, training logs, structure, etc.)
print("+++++++++++++++++ Describe +++++++++++++++++++")
print(model.describe())

# Evaluate a model (e.g. roc, accuracy, confusion matrix, confidence intervals)
print("+++++++++++++++++ Evaluate +++++++++++++++++++")
model.evaluate(test_ds)

# Generate predictions
print("+++++++++++++++++  +++++++++++++++++++")
print(model.predict(test_ds))

# Analyse a model (e.g. partial dependence plot, variable importance)
print(model.analyze(test_ds))

# Benchmark the inference speed of a model
print(model.benchmark(test_ds))

#plt(model.plot_tree())
#plt.savefig('tree.png')
start_time = time.time()
model.predict(test_ds)
end_time = time.time()
prediction_time = end_time - start_time
print(f"Prediction time: {prediction_time} seconds")
model.save("my_RF_model")


# Save Model 
# https://ydf.readthedocs.io/en/latest/tutorial/tf_serving/

#model.to_tensorflow_saved_model("tf_model", mode="tf")

#model.export('model') 

#converter = tf.lite.TFLiteConverter.from_saved_model('model') 

#snort_model = converter.convert() 

#with open('snort.model', 'wb') as f: 
 #   f.write(snort_model) 

#loaded_model = ydf.load_model("my_RF_model")Teh 
