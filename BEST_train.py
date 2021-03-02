import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy.io as scio
import h5py
import numpy as np
import math
import time
import os

dtype = torch.float
device_data = torch.device("cuda:0")
device_train = torch.device("cuda:0")
#device_train = torch.device("cpu")
device_test = torch.device("cuda:0")

TrainingDataSize = 400000
TestingDataSize = 100000
BatchSize = 2000
BatchEnable = True
EpochNum = 501
TestInterval = 10
lr = [1e-4]
lr_decay_step = 50
lr_decay_gamma = 0.8
beta_Constrain = 1e-3
beta_Smooth = 7e-6
TFNum = 16

StartWL = 400
EndWL = 701
Resolution = 1
WL = np.arange(StartWL, EndWL, Resolution)
SpectralSliceNum = WL.size
LayerNum = 500

path_data = 'E:/BEST/data/'  # 远程服务器的数据集路径
name_dataset = 'Specs_general_passive.mat'
name_tfs = 'tfs_general_passive.mat'

specs_train = torch.zeros([TrainingDataSize, SpectralSliceNum], device=device_data, dtype=dtype)
tf_train = torch.zeros([TrainingDataSize, TFNum], device=device_data, dtype=dtype)
specs_test = torch.zeros([TestingDataSize, SpectralSliceNum], device=device_test, dtype=dtype)
tf_test = torch.zeros([TestingDataSize, TFNum], device=device_test, dtype=dtype)

data = h5py.File(path_data + name_dataset, 'r')
Specs_all = np.array(data['Specs_norm2'])
#print(Specs_all.shape)
specs_test[0:TestingDataSize, :] = torch.tensor(Specs_all[: TestingDataSize, :])
specs_train[0:TrainingDataSize, :] = torch.tensor(Specs_all[TestingDataSize:TrainingDataSize+TestingDataSize, :])
data.close()
del Specs_all, data
#print(specs_train.shape)
#print(specs_test.shape)

data = h5py.File(path_data + name_tfs, 'r')
Specs_all = np.array(data['tfs_norm2'])
tf_test[0:TestingDataSize, :] = torch.tensor(Specs_all[0:  TestingDataSize, :])
tf_train[0:TrainingDataSize, :] = torch.tensor(Specs_all[TestingDataSize:TrainingDataSize+TestingDataSize, :])
data.close()
del Specs_all, data




assert SpectralSliceNum == WL.size

MatchLossFcn = nn.MSELoss(reduction='mean')


class hsnetLoss(nn.Module):
    def __init__(self):
        super(hsnetLoss, self).__init__()

    def forward(self, t1, t2):
        MatchLoss = MatchLossFcn(t1, t2)

        return MatchLoss


LossFcn = hsnetLoss()

for k in range(len(lr)):

    folder_name = time.strftime("%Y%m%d_%H%M%S", time.localtime()) + '_TFNum=' + str(TFNum) + '_lr=' + str(lr)
    path = 'nets/' + folder_name + '/'

    hsnet = nn.Sequential()

    hsnet.add_module('Linear1', nn.Linear(TFNum, LayerNum))
    hsnet.add_module('LReLU1', nn.LeakyReLU())
    hsnet.add_module('Linear2', nn.Linear(LayerNum, LayerNum))
    hsnet.add_module('LReLU2', nn.LeakyReLU())
    hsnet.add_module('Linear3', nn.Linear(LayerNum, LayerNum))
    hsnet.add_module('LReLU3', nn.LeakyReLU())
    hsnet.add_module('Linear4', nn.Linear(LayerNum, LayerNum))
    hsnet.add_module('LReLU4', nn.LeakyReLU())
    hsnet.add_module('Linear5', nn.Linear(LayerNum, LayerNum))
    hsnet.add_module('LReLU5', nn.LeakyReLU())
    hsnet.add_module('Linear6', nn.Linear(LayerNum, SpectralSliceNum))
    hsnet = hsnet.to(device_train)

    hsnetParams = hsnet.named_parameters()



    optimizer = torch.optim.Adam(hsnet.parameters(), lr=lr[k])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_gamma)

    loss = torch.tensor([0], device=device_train)
    loss_train = torch.zeros(math.ceil(EpochNum / TestInterval))
    loss_test = torch.zeros(math.ceil(EpochNum / TestInterval))

    os.makedirs(path, exist_ok=True)
    log_file = open(path + 'TrainingLog.txt', 'w+')
    print('TrainingDataSize: ', TrainingDataSize, '| LayerNum:', LayerNum,
          '| Dataset:', name_dataset,'| tfsset:', name_tfs, file=log_file)

    time_start = time.time()
    time_epoch0 = time_start
    tf_noise = torch.zeros([BatchSize, TFNum], device=device_data, dtype=dtype)
    for epoch in range(EpochNum):

        for i in range(0, TrainingDataSize // BatchSize):
            Specs_batch = specs_train[i * BatchSize: i * BatchSize + BatchSize, :].to(device_train)
            tf_batch = tf_train[i * BatchSize: i * BatchSize + BatchSize, :].to(device_train)
            Output_pred = hsnet(tf_batch)
            hsnetParams = hsnet.named_parameters()
            loss = LossFcn(Specs_batch, Output_pred)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        scheduler.step()
        if epoch % TestInterval == 0:
            hsnet.to(device_test)
            out_test_pred = hsnet(tf_test)
            hsnet.to(device_train)
            loss_train[epoch // TestInterval] = loss.data
            loss_t = MatchLossFcn(specs_test, out_test_pred)
            loss_test[epoch // TestInterval] = loss_t.data
            if epoch == 0:
                time_epoch0 = time.time()
                time_remain = (time_epoch0 - time_start) * EpochNum
            else:
                time_remain = (time.time() - time_epoch0) / epoch * (EpochNum - epoch)
            print('Epoch: ', epoch, '| train loss: %.10f' % loss.item(), '| test loss: %.10f' % loss_t.item(),
                  '| learn rate: %.8f' % scheduler.get_lr()[0], '| remaining time: %.0fs (to %s)'
                  % (time_remain, time.strftime('%H:%M:%S', time.localtime(time.time() + time_remain))))
            print('Epoch: ', epoch, '| train loss: %.10f' % loss.item(), '| test loss: %.10f' % loss_t.item(),
                  '| learn rate: %.8f' % scheduler.get_lr()[0], file=log_file)

                # plt.pause(0.001)
                # plt.show()
    time_end = time.time()
    time_total = time_end - time_start
    m, s = divmod(time_total, 60)
    h, m = divmod(m, 60)
    print('训练用时：%.0fs (%dh%02dm%02ds)' % (time_total, h, m, s))
    print('训练用时：%.0fs (%dh%02dm%02ds)' % (time_total, h, m, s), file=log_file)

    hsnet.eval()
    torch.save(hsnet, path + 'hsnet.pkl')
    hsnet.to(device_test)

    hsnetParams = hsnet.named_parameters()

    # plt.show()
    Output_temp = hsnet(tf_train[0, :].to(device_test).unsqueeze(0)).squeeze(0)  # 只有一组数据，使用unsqueeze将其增加一维以使BatchNorm不报错
    FigureTrainLoss = MatchLossFcn(specs_train[0, :].to(device_test), Output_temp)
    plt.figure()
    plt.plot(WL, specs_train[0, :].cpu().numpy())
    plt.plot(WL, Output_temp.detach().cpu().numpy())
    plt.legend(['GT', 'pred'], loc='upper right')
    plt.savefig(path + 'train')
    # plt.show()
    Output_temp = hsnet(tf_test[0, :].to(device_test).unsqueeze(0)).squeeze(0)  # 只有一组数据，使用unsqueeze将其增加一维以使BatchNorm不报错
    FigureTestLoss = MatchLossFcn(specs_test[0, :].to(device_test), Output_temp)
    plt.figure()
    plt.plot(WL, specs_test[0, :].cpu().numpy())
    plt.plot(WL, Output_temp.detach().cpu().numpy())
    plt.legend(['GT', 'pred'], loc='upper right')
    plt.savefig(path + 'test')
    # plt.show()

    print('Training finished!',
          '| loss in figure \'train.png\': %.10f' % FigureTrainLoss.data.item(),
          '| loss in figure \'test.png\': %.10f' % FigureTestLoss.data.item())
    print('Training finished!',
          '| loss in figure \'train.png\': %.10f' % FigureTrainLoss.data.item(),
          '| loss in figure \'test.png\': %.10f' % FigureTestLoss.data.item(), file=log_file)
    log_file.close()

    plt.figure()
    plt.plot(range(0, EpochNum, TestInterval), loss_train.detach().cpu().numpy())
    plt.plot(range(0, EpochNum, TestInterval), loss_test.detach().cpu().numpy())
    plt.semilogy()
    plt.legend(['Loss_train', 'Loss_test'], loc='upper right')
    plt.savefig(path + 'loss')