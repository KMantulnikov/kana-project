from numpy import loadtxt
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageOps
import pandas as pd
 
model_path = 'D:\Other files\Documents\Python\kana-project\models\model_20022021_uniform_augument1.json'
model_weights_path = 'D:\Other files\Documents\Python\kana-project\models\model_weights_20022021_uniform_augument1.h5'
# load json and create model
json_file = open(os.path.join(model_path), 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(os.path.join(model_weights_path))
print("Loaded model from disk")
loss_function = CategoricalCrossentropy()
optimizer = Adam()

# Compile model
loaded_model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])
 
# evaluate loaded model on test data
# loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# score = loaded_model.evaluate(X, Y, verbose=0)
# print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))


test_image = load_img('data/validation/2460/9321.tif')
test_image = test_image.convert('1')
# # print(test_image.mode)
# # test_image.save('test_png.png')
x_test = img_to_array(test_image)*1
x_test = np.expand_dims(x_test, axis = 0)
# x_test = np.swapaxes(x_test,1,2)
test_images = np.vstack([x_test])

classes = np.argmax(loaded_model.predict(test_images, batch_size=1), axis=-1)

plt.figure()
plt.imshow(x_test[-1])


# predicting images
image_path = 'D:\Other files\Documents\Python\kana-project\data\load_testing\ya.jpg'
# img = load_img(os.path.join(image_path), target_size = (63,64))
# img = img.convert('L')
# img = ImageOps.invert(img)
# x = img_to_array(img)
# x = np.expand_dims(x, axis=0)
# images = np.vstack([x])
# plt.figure()
# plt.imshow(x[-1])
# # classes = model.predict_classes(images, batch_size=10)
# classes_of_loaded = np.argmax(loaded_model.predict(images, batch_size=1), axis=-1)

import skimage.color
import skimage.filters
import skimage.io
import skimage.viewer

sigma = 2


# image_sk = skimage.io.imread(fname=image_path)
image_sk = img_to_array(load_img(os.path.join(image_path), target_size = (63,64)))
# blur and grayscale before thresholding
blur = skimage.color.rgb2gray(image_sk)
blur = skimage.filters.gaussian(blur, sigma=sigma)
t = skimage.filters.threshold_otsu(blur)
# perform inverse binary thresholding
mask = blur < t


x_sk = np.expand_dims(mask, axis=0)
x_sk = np.expand_dims(x_sk, axis=3)*1
images_sk = np.vstack([x_sk])
classes_of_loaded_sk = np.argmax(loaded_model.predict(images_sk, batch_size=1), axis=-1)

plt.figure()
plt.imshow(mask)

class_names = pd.read_csv('labels_23022021.csv')
jis_names = pd.read_csv('ETL8B2C1.csv')
jis_names = jis_names.drop_duplicates(subset = ['JIS Kanji Code'])
print(class_names['0'].iloc[classes_of_loaded_sk[0]])
print(jis_names[jis_names['JIS Kanji Code'] == str(class_names['0'].iloc[classes_of_loaded_sk[0]])])
