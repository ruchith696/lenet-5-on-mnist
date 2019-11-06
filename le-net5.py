# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 10:11:54 2019

@author: ruchi
"""

import mnist 
import numpy as np 
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D, AveragePooling2D
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.callbacks import TensorBoard
from time import time



train_images = np.array(mnist.train_images())
train_labels = np.array(mnist.train_labels())

test_images = np.array(mnist.test_images())
test_labels = np.array(mnist.test_labels())

train_images = train_images.reshape((train_images.shape[0],28,28,1))
test_images = test_images.reshape((test_images.shape[0],28,28,1))

train_images = np.pad(train_images,((0,0), (2,2),(2,2),(0,0)), 'constant' )
test_images = np.pad(test_images,((0,0), (2,2),(2,2),(0,0)), 'constant' )

train_features,val_features,train_labels,val_labels=train_test_split(train_images, train_labels, test_size=0.2, random_state=0) 

model = Sequential([
        Conv2D(filters=6, kernel_size=(3, 3), activation='relu', input_shape=(32,32,1)),
        AveragePooling2D(),
        Conv2D(filters = 16, kernel_size = (3,3), activation ='relu'),
        AveragePooling2D(),
        Flatten(),
        Dense(units = 120, activation = 'relu'),
        Dense(units = 84 , activation = 'relu' ),
        Dense(units = 10 , activation = 'softmax')])

model.summary()

model.compile(loss= 'categorical_crossentropy', optimizer= 'adam', metrics=['accuracy'])

EPOCHS = 15
BATCH_SIZE = 128

X_train, y_train = train_features, to_categorical(train_labels)
X_validation, y_validation = val_features, to_categorical(val_labels)

train_generator = ImageDataGenerator().flow(X_train, y_train, batch_size=BATCH_SIZE)
validation_generator = ImageDataGenerator().flow(X_validation, y_validation, batch_size=BATCH_SIZE)

print('# of training images:', train_features.shape[0])
print('# of validation images:', val_features.shape[0])

steps_per_epoch = X_train.shape[0]//BATCH_SIZE
validation_steps = X_validation.shape[0]//BATCH_SIZE

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))
model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=EPOCHS, 
                    validation_data=validation_generator, validation_steps=validation_steps, 
                    shuffle=True, callbacks=[tensorboard])

#model.fit(train_images, to_categorical(train_labels) , batch_size = 128, epochs= 10, validation_split =0.2, shuffle=True)

score = model.evaluate(test_images, to_categorical(test_labels))
print('Test loss:', score[0])
print('Test accuracy:', score[1])