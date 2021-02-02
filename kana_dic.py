import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import random
import time
import subprocess
from PIL import Image

# kana_dic = {'hiragana': {'ka': {'image': img, 'audio': 'n/a'}}}
kana_dic = {'hiragana': {}, 'katakana': {}}
for file in os.listdir('./hiragana/images/'):
    # image = mpimg.imread('./hiragana/images/' + file)
    image = Image.open('./hiragana/images/' + file).convert('LA')
    key = file.split('.')[0]
    kana_dic['hiragana'].update({key: {'image': image, 'audio': 'N/A'}})
    image.close()
    print(file.split('.')[0])


kana_key_seq = random.sample(kana_dic['hiragana'].keys(), 3)

print('Showing sequence')


for kana_key in kana_key_seq:
    print('Showing {0}.'.format(kana_key))

    # plt.figure()
    # plt.imshow(kana_dic['hiragana'][kana_key]['image'])
    # plt.show()
    # os.system('pkill eog')
    # time.sleep(2)
    # plt.close()
    

# plt.ion() # turn on interactive mode
# for kana_key in kana_key_seq:
#     plt.figure()
#     plt.imshow(kana_dic['hiragana'][kana_key]['image'])
#     plt.show()
#     _ = input("Press [enter] to continue.")