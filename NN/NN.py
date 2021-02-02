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

model = Sequential()

model.add(Input((None, 64,64)))

model.add(Conv2D(64, 3, padding = 'same', kernel_initializer = HeNormal()))
model.add(BatchNormalization())
model.add(ReLU())
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2, 2), strides = 2))

model.add(Conv2D(128, 3, padding = 'same', kernel_initializer = HeNormal()))
model.add(BatchNormalization())
model.add(ReLU())
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2, 2), strides = 2))

model.add(Conv2D(192, 3, padding = 'same', kernel_initializer = HeNormal()))
model.add(BatchNormalization())
model.add(ReLU())
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2, 2), strides = 2))

model.add(Conv2D(256, 3, padding = 'same', kernel_initializer = HeNormal()))
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