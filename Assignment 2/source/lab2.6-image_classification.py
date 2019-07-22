import keras
import numpy
from keras.datasets import reuters
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Dropout, Activation, Flatten, K
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.utils import np_utils
from keras.constraints import maxnorm
from keras_preprocessing import sequence
from keras.preprocessing import text
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import pandas as pd
from keras.datasets import fashion_mnist
import numpy as np
from sklearn.model_selection import train_test_split
from IPython import config


(X_train, y_train), (X_test, y_test)= fashion_mnist.load_data()

# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
x_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255

model = Sequential()
model.add(Conv2D(32, kernel_size=(5,5), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(5,5), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10,activation='softmax'))

#Compile the model

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# train the model

model.fit(x=x_train,y=y_train, epochs=2)
# test the model
test_error_rate = model.evaluate(x_test, y_test, verbose=0)
print("The mean squared error (MSE) for the test data set is: {}".format(test_error_rate))
scores = model.evaluate(x_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])
image_index = 7777 # You may select anything up to 60,000
print(y_train[image_index]) # The label is 8
plt.imshow(X_train[image_index], cmap='Greys')
plt.show()