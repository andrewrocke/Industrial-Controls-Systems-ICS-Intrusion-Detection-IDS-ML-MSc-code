
import numpy as np
import pandas as pd




ds = pd.read_csv("ML Test/output_left_singlelabel_add_del_NST.csv")
print("total dataframe :",ds.shape)
print(list(ds))
print(ds.head())
#X=ds['sAddress','rAddress']
#y= ds['IT_B_Label']

#X_train, X_test, y_train,  y_test = train_test_split(X,y ,  
                          #random_state=104, 
                          #train_size=0.8, shuffle=True)

mask = np.random.rand(len(ds)) < 0.8
train_ds = ds[mask] 
test_ds = ds[~mask] 

train_ds.shape, test_ds.shape 

train_ds.to_csv("train_ds.csv")

test_ds.to_csv("test_ds.csv")