import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import time

start = time.time()
# your code...

# load dataset
dataframe = pd.read_csv("output_left_LSTM.csv", header=None)
#dataframe = pd.read_csv("sonar.csv", header=None)
dataset = dataframe.values
# split into input (X) and output (Y) variables
X = dataset[:,0:51].astype(float)
Y = dataset[:,51]

# Option 2: Impute NaN with the mean of the column
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean') # You can use other strategies like 'median'
X = imputer.fit_transform(X)

def create_baseline(epochs, batch_size, verbose):
	# create model
	model = Sequential()
	model.add(Dense(25, input_shape=(51,), activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# evaluate baseline model with standardized dataset
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(model=create_baseline, epochs=100, batch_size=5, verbose=2)))
pipeline = Pipeline(estimators)

model = create_baseline(epochs=100, batch_size=5, verbose=2)
estimator = KerasClassifier(model)
#kfold = StratifiedKFold(n_splits=10, shuffle=False)
#results = cross_val_score(estimator, X, Y)
#print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

predictions = model.predict(X)  # Assuming 'model' is your trained model
predicted_classes = (predictions > 0.5).astype(int) # Convert probabilities to class labels (0 or 1)

accuracy = accuracy_score(Y, predicted_classes)
precision = precision_score(Y, predicted_classes)
recall = recall_score(Y, predicted_classes)
f1 = f1_score(Y, predicted_classes)


print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')

#
# Save Model
#

export_archive = tf.keras.export.ExportArchive()
export_archive.track(model)
export_archive.add_endpoint(
    name='serve',
    fn=model.call,
    input_signature=[tf.TensorSpec(shape=(1, 51), dtype=tf.float32)],
)
export_archive.write_out('model')

converter = tf.lite.TFLiteConverter.from_saved_model('model')

lclassifier2_model = converter.convert()

with open('lclassifier2.model', 'wb') as f:
    f.write(lclassifier2_model)




end = time.time()
print(end - start) # time in seconds
