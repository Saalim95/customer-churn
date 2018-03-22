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

classifier = Sequential()
# Line Below is First Hidden Layer; 6nodes; 11 features(input nodes);initailizaiton of weights 'uniform' (random n small);
classifier.add(Dense(6, input_dim=11 ,kernel_initializer='uniform', activation='relu')) 
classifier.add(Dense(6,kernel_initializer='uniform',activation='relu')) # second Hidden Layer
classifier.add(Dense(1,kernel_initializer='uniform',activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(X_train, y_train, batch_size=10, nb_epoch=20)

y_pred = classifier.predict(X_test) # probablity customer leaves bank
y_pred = (y_pred>0.5) # True-leaves/false-dont leave ; thresh 0.5

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

new_sample = sc.transform(np.array([[0.0 ,0,600,1,40,3,60000,2,1,1,50000]])) #scaled
new_sample_pred = classifier.predict(new_sample)
new_sample_pred = (new_sample_pred>0.5)





