anger=0
disgust=1
fear=2
happy=3
sad=4
surprise=5
neutral=6

import itertools
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
# from keras import models
# from keras import layers
# from keras import optimizers
# from keras import Sequential
# from keras import regularizers
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import classification_report, confusion_matrix
# from keras.applications import VGG16, VGG19
# from keras.layers import Flatten, Dropout, Reshape
# from keras.callbacks import EarlyStopping
# from keras.layers.convolutional import Conv2D, MaxPooling2D
# from keras.models import model_from_json

import pandas as pd
train_df=pd.read_csv("train.csv")
test_df=pd.read_csv("test.csv")

print(train_df.head())

i = 2568
list_pixels = np.array([int(string) for string in train_df.iloc[i,1].split(' ')])
pixeled_image = list_pixels.reshape(48,48)
plt.imshow(pixeled_image, cmap='gray');

x_train = []
for i in range(len(train_df)):
  list_pixels = [int(string) for string in train_df.iloc[i,1].split(' ')]
  x_train.append(list_pixels)
x = np.array(x_train)
y = train_df.iloc[:,0]

from keras.utils import to_categorical
y_cat = to_categorical(np.array(y))

x_train1, x_test, y_train1, y_test =  train_test_split(x, y_cat, random_state = 123, test_size = 0.10)

x_train, x_val, y_train, y_val = train_test_split(x_train1, y_train1, random_state = 123, test_size = 0.20)

print(x_train.shape)
print(x_test.shape) 
print(y_train.shape) 
print(y_test.shape) 
print(x_val.shape)
print(y_val.shape) 

x_train = []
for i in range(len(train_df)):
  # convert string of pixels to list of integers and standarize data
  list_pixels = [int(string)/255 for string in train_df.iloc[i,1].split(' ')]
  # reshape 1D to 2D matrix
  pixeled_image = np.array(list_pixels).reshape(48,48)
  x_train.append(pixeled_image)
  
x = np.array(x_train)
y = train_df.iloc[:,0]

from keras.utils import to_categorical
y = to_categorical(np.array(y))

print(x.shape)
y.shape 

x_train1, x_test, y_train1, y_test =  train_test_split(x, y,  random_state = 123, test_size = 0.10)
x_train, x_val, y_train, y_val = train_test_split(x_train1, y_train1, random_state = 123, test_size = 0.20)

print(x_train.shape) 
print(x_test.shape) 
print(y_train.shape) 
print(y_test.shape) 
print(x_val.shape) 
print(y_val.shape)

x_train_reshaped = np.expand_dims(x_train, axis = 3)
x_val_reshaped = np.expand_dims(x_val, axis = 3)
x_test_reshaped = np.expand_dims(x_test, axis = 3)

x_train_reshaped = np.repeat(x_train_reshaped, 3, axis=3)
x_val_reshaped = np.repeat(x_val_reshaped, 3, axis=3)
x_test_reshaped = np.repeat(x_test_reshaped, 3, axis=3)

print(x_train_reshaped.shape)
print(x_val_reshaped.shape) 
print(x_test_reshaped.shape)

y_test.shape

np.random.seed(123)
cnn_base = tf.keras.applications.VGG16(include_top= False, weights='imagenet', input_shape = (48,48,3))
cnn_base.trainable = False

model = tf.keras.Sequential()

model.add(cnn_base)
# model_cnn.add(layers.Flatten())
model.add(tf.keras.layers.GlobalAveragePooling2D())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(7, activation='softmax')) # 7 classifications
model.summary()

model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])

history = model.fit(x_train_reshaped,y_train,epochs=15, batch_size=50, validation_data=(x_val_reshaped, y_val))

results_train_cnn = model.evaluate(x_test_reshaped, y_test)

model_json = model.to_json()
with open("facial_expression_model_structure.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("facial_expression_model_weights.h5")
print("Saved model")