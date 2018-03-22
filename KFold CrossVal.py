import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:13].values # cols 3 to 12
y = dataset.iloc[:,13].values

# Label Encoder => Label each category as numeric value
# One Hot encoder => to create dumy variable ;Binary  
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
le = LabelEncoder()
X[:, 1] = le.fit_transform(X[:,1]) # Geography
X[:, 2] = le.fit_transform(X[:,2]) # Gender
#applying OHE only on Geog.(3 options); not on gender(only 2 M/F) to avoid fall into dummy variable trap  
onehotencoder = OneHotEncoder( categorical_features = [1]) #create dummy variable
X = onehotencoder.fit_transform(X).toarray()
# 3 dummy variables created[col 1,2,3] for Geog. ; we remove first dummmy [col1/index0] (dummy Var trap)
# Note: A independent variable( feature) with N categories needs  N-1 dummy variables(features)
X = X[:, 1:]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2, random_state= 0)

# FEATURE SCALING
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense

## EVALUATING MODEL USING K-FOLD CROSS VALID.(inside Keras)
from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
cvscores = []
for train, test in kfold.split(X_train, y_train):
  # create model
	model = Sequential()
	model.add(Dense(6, input_dim=11, activation='relu'))
	model.add(Dense(6, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))

	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.fit(X_train[train], y_train[train], epochs=20, batch_size=10, verbose=0)
	# evaluate the model
	scores = model.evaluate(X_train[test], y_train[test], verbose=0)
	print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
	cvscores.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))



