import ydf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split




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
learner = ydf.RandomForestLearner(label="IT_B_Label",
                                  compute_oob_performances=True)

model = learner.train(train_ds)

oob_evaluations = model.out_of_bag_evaluations()
oob_evaluations[-1].evaluation
print("+++++++++++++++++ RF Model +++++++++++++++++++")
print(model.evaluate(test_ds))
#print(model.benchmark(test_ds))
#model.analyze(test_ds, sampling=0.1)
model.save("my_RF_model")

loaded_model = ydf.load_model("my_RF_model")