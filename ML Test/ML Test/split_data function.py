
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split




ds = pd.read_csv("ML Test/output_left_singlelabel_add_del_NST.csv")
print("total dataframe :",ds.shape)
print(list(ds))
print(ds.head())
#X=ds['sAddress','rAddress']
#y= ds['IT_B_Label']
data = ds.values
#ds['sAddress'] = ipaddress.ip_address(ds['sAddress'])
X, y = data[:,:-1], data[:, -1]

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
GMB_train_ds = pd.DataFrame(list(ds))
print("GMB_train_ds", GMB_train_ds.head())



#X_train, X_test, y_train,  y_test = train_test_split(X,y ,  
                          #random_state=104, 
                          #train_size=0.8, shuffle=True)

#mask = np.random.rand(len(ds)) < 0.8
#train_ds = ds[mask] 
#test_ds = ds[~mask] 

X_train_ds.shape, X_test_ds.shape
y_train_ds.shape, y_test_ds.shape 

X_train_ds.to_csv("X_train_ds.csv")
X_test_ds.to_csv("X_test_ds.csv")
y_train_ds.to_csv("y_train_ds.csv")
y_test_ds.to_csv("y_test_ds.csv")