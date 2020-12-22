import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.image as img
import scipy.io as scio
import h5py
import numpy as np
import math
import time
import os

path = 'E:/DeepLearing/test/data/image/20201205/3/'
image_data = np.zeros((480, 640, 16))
specimage_data = np.zeros((480, 640, 301))
for i in range(0, 16):
    image = np.zeros((480, 640))
    #image_bg = img.imread(path + '17_1.png') + img.imread(path + '17_2.png') + img.imread(path + '17_3.png')
    for j in range(0, 1):
        image_input = img.imread(path + str(i+1) + '_' + str(j+1) + '.png')
        image = image + image_input

    image_data[:, :, i] = image/1 #- image_bg/3
image_data[image_data<0] = 0
print(image_data.shape)
print(image_data[1, 1, 1])

net_path = 'nets/hsnet/20201123_191915_TFNum=16_lr=[0.0001]/hsnet.pkl'
net = torch.load(net_path)
time_start = time.time()
for i in range(0, 480):
    for j in range(0, 640):
        tf = image_data[i, j, :]
        maxtf = max(tf)
        tf = tf/maxtf
        tf = torch.tensor(tf, dtype=torch.float)

        spec = net(tf)

       # print(maxtf)
        spec = spec*maxtf
        specimage_data[i, j, :] = spec.detach().numpy()
time_end = time.time()
ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
scio.savemat(path + ts + '.mat', {'SpecImageData': specimage_data})
time_total = time_end - time_start
m, s = divmod(time_total, 60)
h, m = divmod(m, 60)
print('计算用时：%.0fs (%dh%02dm%02ds)' % (time_total, h, m, s))