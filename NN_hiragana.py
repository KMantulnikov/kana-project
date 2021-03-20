import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

import h5py

from sklearn.model_selection import train_test_split

import tensorflow as tf
print(tf.__version__)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, MaxPooling2D, Flatten, Activation, BatchNormalization, SpatialDropout2D
from tensorflow.keras.initializers import HeUniform, HeNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping

# loading dataset
examples_data = h5py.File('data_23022021.h5', 'r')
X_0 = np.array(examples_data.get('X_training'))
# X_0 = np.swapaxes(X_0, 1, 2)  
X_test = np.array(examples_data.get('X_validation'))
# X_test = np.swapaxes(X_test, 1, 2)
Y_0 = np.array(examples_data.get('Y_training'))
Y_test = np.array(examples_data.get('Y_validation'))
examples_data.close()
# encode class values as integers
# Y_training_hot = to_categorical(Y_training, 75)
Y_test_hot = to_categorical(Y_test, 75)


X_train, X_val, Y_train, Y_val = train_test_split(X_0, Y_0, test_size=0.2)

Y_train_hot = to_categorical(Y_train, 75)
Y_val_hot = to_categorical(Y_val, 75)

# learning rate schedule
def step_decay(epoch):
    initial_lrate = 0.0001
    drop = 0.1
    epochs_drop = 20.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    print(lrate)
    return lrate

# create model
def get_model():
    kernel_initializer = HeUniform()
    model = Sequential()
    
    # model.add(Input((X_training.shape[1:])))
    
    model.add(Conv2D(64, (3,3), input_shape = X_train.shape[1:], padding = 'same', kernel_initializer = kernel_initializer))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2, 2), strides = 2, padding = 'same'))
    
    
    model.add(Conv2D(128, (3,3), padding = 'same', kernel_initializer = kernel_initializer))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2, 2), strides = 2, padding = 'same'))
    
    
    model.add(Conv2D(192, (3,3), padding = 'same', kernel_initializer = kernel_initializer))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2, 2), strides = 2, padding = 'same'))
    
    
    model.add(Conv2D(256, (3,3), padding = 'same', kernel_initializer = kernel_initializer))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling2D(pool_size=(2, 2), strides = 2, padding = 'same'))
    
    
    model.add(Flatten())
    
    model.add(Dense(1024, kernel_initializer = kernel_initializer))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(1024, kernel_initializer = kernel_initializer))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Dense(75, activation = 'softmax', kernel_initializer = kernel_initializer))
    
    loss_function = CategoricalCrossentropy()
    optimizer = Adam()
    # tf.keras.metrics.CategoricalAccuracy()
    
    # Compile model
    model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])
    return model


model = get_model()
model.summary()

# define callbacks
early = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1, mode='auto')
lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate, early]

# TRAINING_DIR = 'data/training'
# train_datagen = ImageDataGenerator()

# # TRAIN GENERATOR.
# train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
#                                                     # batch_size=16,
#                                                     class_mode='categorical',
#                                                     target_size=(63, 64))

# VALIDATION_DIR = 'data/validation'
# validation_datagen = ImageDataGenerator()


# # VALIDATION GENERATOR.
# validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
#                                                            # batch_size=16,
#                                                           class_mode  = 'categorical',
#                                                           target_size = (63, 64))


train_datagen = ImageDataGenerator(rotation_range=5)
validation_datagen = ImageDataGenerator(rotation_range=5)

train_generator = train_datagen.flow(X_train, Y_train_hot, batch_size=16, shuffle=True)

valid_generator = validation_datagen.flow(X_val, Y_val_hot, batch_size=16, shuffle=True)

print(train_generator.n)
print(valid_generator.n)

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size



history = model.fit(train_generator,\
          steps_per_epoch=STEP_SIZE_TRAIN,\
          validation_data=valid_generator,\
          validation_steps=STEP_SIZE_VALID,\
          epochs=400,\
          callbacks=callbacks_list)

# history = model.fit(x = X_training, y = Y_training_hot, epochs=400, batch_size=16, verbose=1, validation_data=(X_validation, Y_validation_hot),\
#                     callbacks=callbacks_list)


# history = model.fit(x = X_training, y = Y_training_hot, epochs=400, batch_size=16, verbose=1, validation_data=(X_validation, Y_validation_hot),\
#                     callbacks=callbacks_list)
    
# history = model.fit(train_generator, epochs=20, batch_size=16, verbose=1, validation_data=validation_generator)
   

# serialize model to JSON
model_json = model.to_json()
with open("models/model_20022021_uniform_augument1.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("models/model_weights_20022021_uniform_augument1.h5")
print("Saved model to disk")    
 
model.evaluate(X_test, Y_test_hot, batch_size=16, verbose=1)    
    
# test_image = load_img('data/validation/2460/9321.tif')
# test_image = test_image.convert('1')
# # print(test_image.mode)
# # test_image.save('test_png.png')
# x_test = img_to_array(test_image)*1
# x_test = np.expand_dims(x_test, axis = 0)
# x_test = np.swapaxes(x_test,1,2)
# test_images = np.vstack([x_test])

# classes = np.argmax(model.predict(test_images, batch_size=10), axis=-1)


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
plt.plot(acc, 'r', label='Training accuracy')
plt.plot(val_acc, 'b', label='Validation accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend()
plt.title('Accuracy')


# plt.figure()
# plt.plot(val_acc, 'b')
# plt.title('Val accuracy')

plt.figure()
plt.plot(loss, 'r', label='Training loss')
plt.plot(val_loss, 'b', label='Validation loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.title('Loss')

# plt.figure()
# plt.plot(val_loss, 'b')
# plt.title('Val loss')

# import pickle
# with open('trainHistoryDict', 'wb') as file_pi:
#     pickle.dump(history.history, file_pi)