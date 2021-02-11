import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import h5py

import tensorflow as tf
print(tf.__version__)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, MaxPooling2D, Flatten, ReLU, BatchNormalization
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = Sequential()

# model.add(Input((64,63)))

model.add(Conv2D(64, 3, padding = 'same', kernel_initializer = HeNormal(), input_shape=(64,63,3)))
model.add(BatchNormalization())
model.add(ReLU())
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2, 2), strides = 2))

model.add(Conv2D(128, (3,3), padding = 'same', kernel_initializer = HeNormal()))
model.add(BatchNormalization())
model.add(ReLU())
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2, 2), strides = 2))

model.add(Conv2D(192, (3,3), padding = 'same', kernel_initializer = HeNormal()))
model.add(BatchNormalization())
model.add(ReLU())
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2, 2), strides = 2))

model.add(Conv2D(256, (3,3), padding = 'same', kernel_initializer = HeNormal()))
model.add(BatchNormalization())
model.add(ReLU())
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2, 2), strides = 2))

model.add(Flatten())

model.add(Dense(1024, kernel_initializer = HeNormal()))
model.add(BatchNormalization())
model.add(ReLU())
model.add(Dropout(0.5))

model.add(Dense(1024, kernel_initializer = HeNormal()))
model.add(BatchNormalization())
model.add(ReLU())

model.add(Dense(75, activation = 'softmax', kernel_initializer = HeNormal()))

loss_function = CategoricalCrossentropy()
optimizer = Adam(learning_rate=0.0001)
# tf.keras.metrics.CategoricalAccuracy()
model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])
model.summary()


TRAINING_DIR = 'data/training'
train_datagen = ImageDataGenerator()

# NOTE: YOU MUST USE A BATCH SIZE OF 10 (batch_size=10) FOR THE 
# TRAIN GENERATOR.
train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                    batch_size=16,
                                                    class_mode='categorical',
                                                    target_size=(64, 63))

VALIDATION_DIR = 'data/validation'
validation_datagen = ImageDataGenerator()

# NOTE: YOU MUST USE A BACTH SIZE OF 10 (batch_size=10) FOR THE 
# VALIDATION GENERATOR.
validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                         batch_size=16,
                                                         class_mode  = 'categorical',
                                                         target_size = (64, 63))

history = model.fit(train_generator, epochs=20, verbose=1, validation_data=validation_generator)



#-----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
#-----------------------------------------------------------
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(len(acc)) # Get number of epochs

#------------------------------------------------
# Plot training and validation accuracy per epoch
#------------------------------------------------
plt.figure()
plt.plot(acc, 'r')


plt.figure()
plt.plot(val_acc, 'b')

plt.figure()
plt.plot(loss, 'r')

plt.figure()
plt.plot(val_loss, 'b')