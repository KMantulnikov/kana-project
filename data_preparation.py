
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


meta_data = pd.read_csv('./ETL8B/ETL8B2C1.csv')
meta_data = meta_data[:12000]
im = Image.open('./ETL8B/ETL8B2C1_00.png')
# a_hira = meta_data[meta_data['JIS Typical Reading'] == 'A.HI']

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
imgplot = plt.imshow(im)
plt.scatter(meta_data['x_0'],meta_data['y_0'])
plt.scatter(meta_data['x_1'],meta_data['y_1'])

for index, row in meta_data.iterrows():
    print(index, row['JIS Typical Reading'], row['x_0'], row['y_0'], row['x_1'], row['y_1'])
    im_crop = im.crop((row['x_0'], row['y_0'], row['x_1'], row['y_1']))
    im_crop.save('cropped_images/' + row['JIS Typical Reading'] + '_' + str(index) + '.png')

# plt.figure('image_crop')
# imgplot = plt.imshow(im_crop)
