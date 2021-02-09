import os
import random
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


meta_data = pd.read_csv('./ETL8B/ETL8B2C1.csv')
meta_data = meta_data[:12000]
# a_hira = meta_data[meta_data['JIS Typical Reading'] == 'A.HI']

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

for name in np.unique(meta_data['JIS Typical Reading']):
    name = name.replace('.','_')
    target_path_training = os.path.join('data/training/' + name)
    target_path_validation = os.path.join('data/validation/' + name)
    if not os.path.exists(target_path_training):
        os.makedirs(target_path_training)
        print('{0} created'.format(target_path_training))
    else:
        print('Directory already exists.')
        
    if not os.path.exists(target_path_validation):
        os.makedirs(target_path_validation)
        print('{0} created'.format(target_path_validation))
    else:
        print('Directory already exists.')

randomlist = np.array(random.sample(range(0, 159), 30))  
n = 0
for index, row in meta_data.iterrows():
    im = images['im_0{0}'.format(n)]
    
    print(index, row['JIS Typical Reading'], row['x_0'], row['y_0'], row['x_1'], row['y_1'])
            
    im_crop = im.crop((row['x_0'], row['y_0'], row['x_1'], row['y_1']))
    
    if index in randomlist:
        im_crop.save('data/validation/' + (row['JIS Typical Reading']).replace('.','_') + '/' + str(index) + '.png')
    else:
        im_crop.save('data/training/' + (row['JIS Typical Reading']).replace('.','_') + '/' + str(index) + '.png')
    
    if index > 0 and index % 160 == 0:
        randomlist += 160
    
    if index >= 2000 and index % 2000 == 0:
        print(index)
        n += 1
        print('Changing sheet to 0{0}'.format(n))
