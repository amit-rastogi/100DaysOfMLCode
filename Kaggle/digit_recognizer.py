"""
Created on Sat Sep 22 15:18:42 2018

@author: Amit Rastogi
"""

import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers as reg
import keras.utils as ku

dataset_train = pd.read_csv('train.csv')
dataset_test = pd.read_csv('test.csv')

train_X = dataset_train.iloc[:,1:dataset_train.shape[1]]
train_X = train_X/255

train_y = dataset_train.iloc[:,0]
binary_train_y = ku.to_categorical(train_y)

test_X = dataset_test.iloc[:,0:dataset_test.shape[1]]
test_X = test_X/255

num_units = train_X.shape[1]

classifier = Sequential()
classifier.add(Dense(units=num_units, kernel_initializer='uniform',
                     activation='relu', input_dim=train_X.shape[1]))                     
classifier.add(Dense(units=num_units, kernel_initializer='uniform',
                     activation='relu'))       
classifier.add(Dense(units=train_y.nunique(), kernel_initializer='uniform',
                     activation='softmax'))
classifier.compile(optimizer='adam', loss='categorical_crossentropy',
                   metrics=['accuracy'])
classifier.fit(train_X, binary_train_y, batch_size=256, nb_epoch=30)


y_pred = classifier.predict(test_X)
y_pred_final = y_pred.argmax(1)
my_submission = pd.DataFrame({'ImageId': np.arange(1,dataset_test.shape[0]+1), 'Label': y_pred_final})
my_submission.to_csv('submission.csv', index=False)
