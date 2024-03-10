#https://www.youtube.com/watch?v=6_2hzRopPbQ
#https://github.com/nicknochnack/Tensorflow-in-10-Minutes/blob/main/Tensorflow%20in%2010.ipynb

############################################
import pandas as pd
import numpy as np
df = pd.read_csv('TensorFlow/data/Churn.csv')
'''
Tensorflow - ValueError: Failed to convert a NumPy array to a Tensor (Unsupported object type float/int)
https://stackoverflow.com/questions/58636087/tensorflow-valueerror-failed-to-convert-a-numpy-array-to-a-tensor-unsupporte
The problem's rooted in using lists as inputs, as opposed to Numpy arrays; Keras/TF doesn't support former. A simple conversion is: x_array = np.asarray(x_list).
'''
X = pd.get_dummies(df.drop(['Churn', 'Customer ID'], axis=1))
cols=len(X.columns)
X.head()
#### 
X = np.asarray(X).astype('float32')
y = df['Churn'].apply(lambda x: 1 if x=='Yes' else 0)
y = np.asarray(y).astype('float32')

##########################################
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

#y_train.head()
##################################

####### 1. Import Dependencies
#### Error on dist utils so install and iport setu[ptools first as disiutils is deprecated]
#### https://stackoverflow.com/questions/77233855/why-did-i-got-an-error-modulenotfounderror-no-module-named-distutils
import setuptools
import tensorflow as tf
import keras
#Import TensorFlow:
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score

#https://www.activestate.com/resources/quick-reads/how-to-install-keras-and-tensorflow/#:~:text=TensorFlow%20and%20Keras%20require%20Python%203.6%2B%20%28Python%203.8,--version%20Output%20should%20be%20similar%20to%3A%20Python%203.8.2
'''
CPU any modern computer can run this version, but it offers the slowest training speeds.
TPU only available currently on Google’s Colaboratory (Colab) platform, Tensor Processing Units (TPUs) offer the highest training speeds.
GPU most high end computers feature a separate Graphics Processing Unit (GPU) from Nvidia or AMD that offer training speeds much faster than CPUs, but not as fast as TPUs. 
'''
####################################
####### 2. Build and Compile Model
model = Sequential()
model.add(Dense(units=32, activation='relu', input_dim=(cols-2)))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))
### compile model
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])


##########################################
#### 3. Fit, Predict and Evaluate
model.fit(X_train, y_train, epochs=200, batch_size=32)
#FAILED HERE




y_hat = model.predict(X_test)
y_hat = [0 if val < 0.5 else 1 for val in y_hat]
accuracy_score(y_test, y_hat)
#0.7814052519517388

###################################################
####4. Saving and Reloading
model.save('tfmodel')
del model 
model = load_model('tfmodel')