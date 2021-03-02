import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.image as img
import scipy.io as scio
import h5py
import numpy as np
import math
import time
import datetime
import os

picnum = 1

path = 'data/image/active/paint/'
image_data = np.zeros((480, 640, 16))
image_data_de = np.zeros((480, 640, 16))
image_data_max = np.zeros((480, 640))
specimage_data = torch.zeros((480, 640, 301), device=torch.device("cuda:0"), dtype = torch.float)
for i in range(0, 16):
    image = np.zeros((480, 640))
    #image_bg = img.imread(path + '17_1.png') + img.imread(path + '17_2.png') + img.imread(path + '17_3.png')
    for j in range(0, picnum):
        image_input = img.imread(path + str(i+1) + '_' + str(j+1) + '.png')
        image = image + image_input

    image_data[:, :, i] = image/picnum #- image_bg/3
image_data[image_data<0] = 0
print(image_data.shape)
print(image_data[1, 1, 1])

net_path = 'nets/active/hsnet.pkl'
net = torch.load(net_path)

for i in range(0, 480):
    for j in range(0, 640):
        tf = image_data[i, j, :]
        #tf = tf - min(tf)
        maxtf = max(tf)
        tf = tf/maxtf
        image_data_max[i, j] = maxtf
        image_data_de[i, j, :] = tf


image_data_tensor = torch.tensor(image_data_de, device=torch.device("cuda:0"), dtype = torch.float)
time_start = datetime.datetime.now()
specimage_data = net(image_data_tensor)

time_end = datetime.datetime.now()
time_total = time_end - time_start
print(time_total)
ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
scio.savemat(path + ts + '.mat', {'SpecImageData': specimage_data.detach().cpu().numpy()})