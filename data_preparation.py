import os
import random
import h5py
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


meta_data = pd.read_csv('./ETL8B/ETL8B2C1.csv')
meta_data = meta_data[:12000]

labels = np.unique(meta_data['JIS Kanji Code'])
labels = np.sort(labels)
labels = pd.DataFrame(labels)
labels['label'] = labels.index
# cur_label = labels[labels[0] == '2444']
# a_hira = meta_data[meta_data['JIS Typical Reading'] == 'TSU.']

images = {
            'im_00': Image.open('./ETL8B/ETL8B2C1_00.png'),
            'im_01': Image.open('./ETL8B/ETL8B2C1_01.png'),
            'im_02': Image.open('./ETL8B/ETL8B2C1_02.png'),
            'im_03': Image.open('./ETL8B/ETL8B2C1_03.png'),
            'im_04': Image.open('./ETL8B/ETL8B2C1_04.png'),
            'im_05': Image.open('./ETL8B/ETL8B2C1_05.png')
            }

x_0 = []
y_0 = []
for row in range(40):
    for column in range(50):
        x_0.append(column*64)
        y_0.append(row*63)

x_0 = np.array(x_0)
y_0 = np.array(y_0)

x_1 = x_0 + 64
y_1 = y_0 + 63

x_0 = np.tile(x_0, 6)
y_0 = np.tile(y_0, 6)

x_1 = np.tile(x_1, 6)
y_1 = np.tile(y_1, 6)

meta_data['x_0'] = x_0
meta_data['y_0'] = y_0

meta_data['x_1'] = x_1
meta_data['y_1'] = y_1


plt.figure('image')
imgplot = plt.imshow(images['im_00'])
plt.scatter(meta_data['x_0'],meta_data['y_0'])
plt.scatter(meta_data['x_1'],meta_data['y_1'])


    
# for name in np.unique(meta_data['JIS Typical Reading']):
#     name = name.replace('.','_')
#     target_path = os.path.join('source/' + name)
#     if not os.path.exists(target_path):
#         os.makedirs(target_path)
#         print('{0} created'.format(target_path))
#     else:
#         print('Directory already exists.')



# for name in np.unique(meta_data['JIS Kanji Code']):
#     name = name.replace('.','_')
    # target_path_training = os.path.join('data/training/' + name)
    # target_path_validation = os.path.join('data/validation/' + name)
    # if not os.path.exists(target_path_training):
    #     os.makedirs(target_path_training)
    #     print('{0} created'.format(target_path_training))
    # else:
    #     print('Directory already exists.')
        
    # if not os.path.exists(target_path_validation):
    #     os.makedirs(target_path_validation)
    #     print('{0} created'.format(target_path_validation))
    # else:
    #     print('Directory already exists.')

X_training = []
Y_training = []
X_validation = []
Y_validation = []

randomlist = np.array(random.sample(range(0, 159), 30))  
n = 0
for index, row in meta_data.iterrows():
    im = images['im_0{0}'.format(n)]
    
    print(index, row['JIS Typical Reading'], row['x_0'], row['y_0'], row['x_1'], row['y_1'])
            
    im_crop = im.crop((row['x_0'], row['y_0'], row['x_1'], row['y_1']))
    
    if index in randomlist:
        cur_label = labels[labels[0] == row['JIS Kanji Code']]
        im_crop_bw = im_crop.convert('L')
        im_crop_array = np.asarray(im_crop)
        im_crop_array = np.expand_dims(im_crop_array, axis = 0)
        # np.append(X_training, im_crop_array, axis = 0)
        X_validation.append(im_crop_array*1) 
        Y_validation.append(cur_label['label'])
        # im_crop_bw.save('data/validation/' + (row['JIS Kanji Code']).replace('.','_') + '/' + str(index) + '.tif')
    else:
        cur_label = labels[labels[0] == row['JIS Kanji Code']]
        # im_crop_bw = im_crop.convert('L')
        im_crop_array = np.asarray(im_crop)
        im_crop_array = np.expand_dims(im_crop_array, axis = 0)
        # np.append(X_training, im_crop_array, axis = 0)
        X_training.append(im_crop_array*1) 
        Y_training.append(cur_label['label'])
        # print(im_crop_array.shape)
        # # im_crop_bw = im_crop.convert('L')
        # im_crop_bw = Image.fromarray(im_crop_array)   
        # print(im_crop_bw.mode)
        # im_crop_bw.save('data/training/' + (row['JIS Kanji Code']).replace('.','_') + '/' + str(index) + '.tif')
    
    if index > 0 and index % 160 == 0:
        randomlist += 160
    
    if (index + 1) >= 2000 and (index + 1) % 2000 == 0:
        print(index)
        n += 1
        print('Changing sheet to 0{0}'.format(n))

X_training = np.array(X_training)
X_training = np.swapaxes(X_training,1,3)
X_training = np.swapaxes(X_training,1,2)
X_validation = np.array(X_validation)
X_validation = np.swapaxes(X_validation,1,3)
X_validation = np.swapaxes(X_validation,1,2)
Y_training = np.array(Y_training)
Y_validation = np.array(Y_validation)

ex_file = h5py.File('data_23022021.h5', 'w')
ex_file.create_dataset('X_training', data=X_training)
ex_file.create_dataset('X_validation', data=X_validation)
ex_file.create_dataset('Y_training', data=Y_training)
ex_file.create_dataset('Y_validation', data=Y_validation)
ex_file.close()

labels.to_csv('labels_23022021.csv')

# plt.figure()
# plt.imshow(X_validation[-2])