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
device_data = torch.device("cpu")
#device_train = torch.device("cuda:0")
device_train = torch.device("cpu")
device_test = torch.device("cpu")

TrainingDataSize = 1000000
TestingDataSize = 10000
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
noise_level = 0.1

StartWL = 400
EndWL = 701
Resolution = 1
WL = np.arange(StartWL, EndWL, Resolution)
SpectralSliceNum = WL.size
Layer1Num = 500
Layer2Num = 500
Layer3Num = 500

path_data = 'E:/DeepLearing/test/data/'  # 远程服务器的数据集路径
# path_data = 'D:/科研/光场成像/Spectral imaging/SpecSamples/data/'
specs_train = torch.zeros([TrainingDataSize, SpectralSliceNum], device=device_data, dtype=dtype)
tf_train = torch.zeros([TrainingDataSize, TFNum], device=device_data, dtype=dtype)
specs_test = torch.zeros([TestingDataSize, SpectralSliceNum], device=device_test, dtype=dtype)
tf_test = torch.zeros([TestingDataSize, TFNum], device=device_test, dtype=dtype)
data = h5py.File(path_data + 'ICVL/SpectralCurves/Specs_ICVLnorm2exp_BP.mat', 'r')
Specs_all = np.array(data['Specs_norm2'])

#np.random.shuffle(Specs_all)
print(Specs_all.shape)
#specs_train[0:TrainingDataSize//2, :] = torch.tensor(Specs_all[0:TrainingDataSize//2, :])
specs_test[0:TestingDataSize//2, 0:151] = torch.tensor(
    Specs_all[: TestingDataSize//2, :])
data = h5py.File(path_data + 'CAVE/SpectralCurves/Specs_CAVEnorm2_BP.mat', 'r')
Specs_all = np.array(data['Specs_norm2']) #/65536
np.random.shuffle(Specs_all)
#specs_train[TrainingDataSize//2:TrainingDataSize, :] = torch.tensor(Specs_all[0:TrainingDataSize//2, :])
specs_test[TestingDataSize//2:TestingDataSize, 0:151] = torch.tensor(
    Specs_all[0: TestingDataSize//2, :])

data.close()
del Specs_all, data

data = h5py.File(path_data + 'Specs_trans2_passive005_led.mat', 'r')
Specs_all = np.array(data['Specs_norm2'])
specs_train[0:TrainingDataSize, :] = torch.tensor(Specs_all[0:TrainingDataSize, :])
data.close()
del Specs_all, data


data = h5py.File(path_data + 'tfs_ICVLnorm2exp_BP.mat', 'r')
Specs_all = np.array(data['tfs_norm2'])

#np.random.shuffle(Specs_all)

#tf_train[0:TrainingDataSize//2, :] = torch.tensor(Specs_all[0:TrainingDataSize//2, [6, 9, 12, 14]])
tf_test[0:TestingDataSize//2, :] = torch.tensor(
    Specs_all[0:  TestingDataSize//2, :])
data = h5py.File(path_data + 'tfs_CAVEnorm2_BP.mat', 'r')
Specs_all = np.array(data['tfs_norm2']) #/65536
#np.random.shuffle(Specs_all)
#tf_train[TrainingDataSize//2:TrainingDataSize, :] = torch.tensor(Specs_all[0:TrainingDataSize//2, [6, 9, 12, 14]])
tf_test[TestingDataSize//2:TestingDataSize, :] = torch.tensor(
    Specs_all[0: TestingDataSize//2, :])

data.close()
del Specs_all, data

data = h5py.File(path_data + 'tfs_trans2_passive005_led.mat', 'r')
Specs_all = np.array(data['tfs_norm2'])
tf_train[0:TrainingDataSize, :] = torch.tensor(Specs_all[0:TrainingDataSize, :])


data.close()
del Specs_all, data
#filterdata = h5py.File(path_data + 'filteredLED.mat', 'r')
#specs_filter = torch.tensor(filterdata['Specs_filteredLED'],device=device_data, dtype=dtype)
#specs_filter = specs_filter[:, 0:TFNum]
print(specs_train.shape)
print(specs_test.shape)
# plt.plot(WL, Specs[0, :].cpu().numpy())
# plt.ylim(0, 1)
# plt.show()
#print(specs_filter.shape)
tf_train=tf_train*1
specs_train=specs_train*1
tf_test=tf_test*1
specs_test=specs_test*1


#filterdata.close()
#del filterdata

assert SpectralSliceNum == WL.size

MatchLossFcn = nn.MSELoss(reduction='mean')


class hsnetLoss(nn.Module):
    def __init__(self):
        super(hsnetLoss, self).__init__()

    def forward(self, t1, t2):
        """
        hsnet的损失函数由三部分组成：
        1.表示拟合误差的MatchLoss
        2.限制HardwareLayer取值在0-1之间的ConstrainLoss
        3.限制HardwareLayer曲线光滑程度的SmoothLoss
        """
        MatchLoss = MatchLossFcn(t1, t2)

        # 直线U形函数，U([delta, 1-delta])=0, x大于1-delta和x小于delta时U为直线上升，U(0)=U(1)=1。
        #delta = 0.01
        #res = torch.max((params - delta) / (-delta), (params + delta - 1) / delta)
       # RangeLoss = torch.mean(torch.max(res, torch.zeros_like(res)))
        # U形函数，U(0.5)=0，U(x)随x与0.5的差呈平方指数上升。
        # 与KL-Loss函数相比，优点为在整个实数域上都有定义，保证了LossFcn在训练过程中始终有梯度。
        # rho = 50  # 控制U形函数的陡峭程度的参数，rho越大越陡峭。最好取rho>>1，此时U(0), U(1)-->1。
        # RangeLoss = torch.mean(torch.exp(rho*(4*(params-0.5)**2-1))-e**(-rho))
        # KL-Loss函数，x=rho时函数为0，x=0, 1时函数为Inf。
        # 训练时容易出现Loss=nan的情况，导致梯度消失无法收敛。
        # rho = 0.5
        # RangeLoss = sum(sum(rho * torch.log(rho / params) + (1-rho) * torch.log((1-rho)/(1-params))))

        # params_scaled = torch.div(params, torch.max(params, 1)[0].unsqueeze(0).T)
       # shift_diff = params - params.roll(1)
        #shift_diff[:, 0] = 0
       # SmoothLoss = torch.norm(shift_diff)

        return MatchLoss #+ beta_Range * RangeLoss + beta_Smooth * SmoothLoss


LossFcn = hsnetLoss()

for k in range(len(lr)):
    # folder_name = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    folder_name = time.strftime("%Y%m%d_%H%M%S", time.localtime()) + '_TFNum=' + str(TFNum) + '_lr=' + str(lr)
    path = 'nets/hsnet/' + folder_name + '/'
   # rnet_path = 'nets/rnet/20200608_215452_Meta_bricks_9^4_interp/rnet.pkl'
   # fnet_path = 'nets/fnet/20200608_213707_Meta_bricks_9^4_interp_drop/fnet.pkl'

    hsnet = nn.Sequential()
 #   hsnet.add_module('HardwareLayer', nn.Linear(SpectralSliceNum, TFNum))
 #  hsnet.add_module('LReLU1', nn.LeakyReLU())
    hsnet.add_module('Linear1', nn.Linear(TFNum, Layer1Num))
    hsnet.add_module('LReLU1', nn.LeakyReLU())
    hsnet.add_module('Linear2', nn.Linear(Layer1Num, Layer2Num))
    hsnet.add_module('LReLU2', nn.LeakyReLU())
    hsnet.add_module('Linear3', nn.Linear(Layer2Num, Layer3Num))
    hsnet.add_module('LReLU3', nn.LeakyReLU())
    hsnet.add_module('Linear4', nn.Linear(Layer3Num, 500))
    hsnet.add_module('LReLU4', nn.LeakyReLU())
    hsnet.add_module('Linear5', nn.Linear(Layer3Num, 500))
    hsnet.add_module('LReLU5', nn.LeakyReLU())
    hsnet.add_module('Linear6', nn.Linear(500, SpectralSliceNum))
    hsnet = hsnet.to(device_train)

    hsnetParams = hsnet.named_parameters()



    optimizer = torch.optim.Adam(hsnet.parameters(), lr=lr[k])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_gamma)

    loss = torch.tensor([0], device=device_train)
    loss_train = torch.zeros(math.ceil(EpochNum / TestInterval))
    loss_test = torch.zeros(math.ceil(EpochNum / TestInterval))

    os.makedirs(path, exist_ok=True)
    os.makedirs(path + 'curves_evolution/', exist_ok=True)
    log_file = open(path + 'TrainingLog.txt', 'w+')
    print('TrainingDataSize: ', TrainingDataSize, '| noise_level:', noise_level, '| Layer1Num:', Layer1Num,
          '| Layer2Num', Layer2Num,'| Layer3Num', Layer3Num, file=log_file)

    time_start = time.time()
    time_epoch0 = time_start
    tf_noise = torch.zeros([BatchSize, TFNum], device=device_data, dtype=dtype)
    for epoch in range(EpochNum):
        #specs_train = specs_train[torch.randperm(TrainingDataSize), :]
        for i in range(0, TrainingDataSize // BatchSize):
            Specs_batch = specs_train[i * BatchSize: i * BatchSize + BatchSize, :].to(device_train)
            #tf_noise = noise_level*(torch.rand([BatchSize, TFNum], device=device_data, dtype=dtype)-0.5)
            #tf_batch = torch.mm(Specs_batch, specs_filter)
            tf_batch = tf_train[i * BatchSize: i * BatchSize + BatchSize, :].to(device_train)
            #if (epoch % 2) == 0:
               # tf_batch = tf_batch + tf_noise
            Output_pred = hsnet(tf_batch)
            hsnetParams = hsnet.named_parameters()
            loss = LossFcn(Specs_batch, Output_pred)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        scheduler.step()
        if epoch % TestInterval == 0:
            hsnet.to(device_test)
            #tf_test = torch.mm(specs_test, specs_filter)
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
            print('Epoch: ', epoch, '| train loss: %.8f' % loss.item(), '| test loss: %.5f' % loss_t.item(),
                  '| learn rate: %.8f' % scheduler.get_lr()[0], '| remaining time: %.0fs (to %s)'
                  % (time_remain, time.strftime('%H:%M:%S', time.localtime(time.time() + time_remain))))
            print('Epoch: ', epoch, '| train loss: %.8f' % loss.item(), '| test loss: %.5f' % loss_t.item(),
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
   # tf_train = torch.mm(specs_train, specs_filter)
    Output_temp = hsnet(tf_train[0, :].to(device_test).unsqueeze(0)).squeeze(0)  # 只有一组数据，使用unsqueeze将其增加一维以使BatchNorm不报错
    FigureTrainLoss = MatchLossFcn(specs_train[0, :].to(device_test), Output_temp)
    plt.figure()
    plt.plot(WL, specs_train[0, :].cpu().numpy())
    plt.plot(WL, Output_temp.detach().cpu().numpy())
   # plt.ylim(0, 1)
    plt.legend(['GT', 'pred'], loc='upper right')
    plt.savefig(path + 'train')
    # plt.show()

   # tf_test = torch.mm(specs_test, specs_filter)
    Output_temp = hsnet(tf_test[0, :].to(device_test).unsqueeze(0)).squeeze(0)  # 只有一组数据，使用unsqueeze将其增加一维以使BatchNorm不报错
    FigureTestLoss = MatchLossFcn(specs_test[0, :].to(device_test), Output_temp)
    plt.figure()
    plt.plot(WL, specs_test[0, :].cpu().numpy())
    plt.plot(WL, Output_temp.detach().cpu().numpy())
   # plt.ylim(0, 1)
    plt.legend(['GT', 'pred'], loc='upper right')
    plt.savefig(path + 'test')
    # plt.show()

    print('Training finished!',
          '| loss in figure \'train.png\': %.8f' % FigureTrainLoss.data.item(),
          '| loss in figure \'test.png\': %.5f' % FigureTestLoss.data.item())
    print('Training finished!',
          '| loss in figure \'train.png\': %.8f' % FigureTrainLoss.data.item(),
          '| loss in figure \'test.png\': %.5f' % FigureTestLoss.data.item(), file=log_file)
    log_file.close()

    plt.figure()
    plt.plot(range(0, EpochNum, TestInterval), loss_train.detach().cpu().numpy())
    plt.plot(range(0, EpochNum, TestInterval), loss_test.detach().cpu().numpy())
    plt.semilogy()
    plt.legend(['Loss_train', 'Loss_test'], loc='upper right')
    plt.savefig(path + 'loss')