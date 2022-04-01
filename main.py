import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
import numpy as np
from spikingjelly.clock_driven import neuron, encoding, functional
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from spikingjelly.datasets import pad_sequence_collate, padded_sequence_mask
from spikingjelly.datasets.n_mnist import NMNIST
from torch.utils.data import Dataset as BaseDataset
from torch.utils.data import DataLoader
import os
import sys

parser = argparse.ArgumentParser(description='spikingjelly LIF MNIST Training')

parser.add_argument('--device', default='cuda:0', help='运行的设备，例如“cpu”或“cuda:0”\n Device, e.g., "cpu" or "cuda:0"')

parser.add_argument('--dataset-dir', default='./', help='保存MNIST数据集的位置，例如“./”\n Root directory for saving MNIST dataset, e.g., "./"')
parser.add_argument('--log-dir', default='./result', help='保存tensorboard日志文件的位置，例如“./”\n Root directory for saving tensorboard logs, e.g., "./"')
parser.add_argument('--model-output-dir', default='./', help='模型保存路径，例如“./”\n Model directory for saving, e.g., "./"')

parser.add_argument('-b', '--batch-size', default=160, type=int, help='Batch 大小，例如“64”\n Batch size, e.g., "64"')
parser.add_argument('-T', '--timesteps', default=100, type=int, dest='T', help='仿真时长，例如“100”\n Simulating timesteps, e.g., "100"')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, metavar='LR', help='学习率，例如“1e-3”\n Learning rate, e.g., "1e-3": ', dest='lr')
parser.add_argument('--tau', default=2.0, type=float, help='LIF神经元的时间常数tau，例如“100.0”\n Membrane time constant, tau, for LIF neurons, e.g., "100.0"')
parser.add_argument('-N', '--epoch', default=100, type=int, help='训练epoch，例如“100”\n Training epoch, e.g., "100"')

from torch.autograd import Variable


class Dataset(BaseDataset):
    """
    For time-base evnet data
    
    """
    
    def __init__(
        self, 
        images_dir,
        train_mode,
        # event_set,
    ):
        # ids = os.listdir(images_dir)
        self.images_dir = images_dir
        event_set = NMNIST(images_dir, train=train_mode, data_type='event')
         # event, label = event_set[0]
        self.input = []
        self.label = []
        
        for i in tqdm(range(len(event_set))):
            event, label = event_set[i]
            len_data = len(event['t'])
            input_array = np.zeros([3, 2048], np.float)
            if len_data > 2048:
                input_array[0, 0:len_data] = event['x'][0:2048]
                input_array[1, 0:len_data] = event['y'][0:2048]
                input_array[2, 0:len_data] = event['t'][0:2048] / 1000 # to ms
            # input_array
            else:
                input_array[0, 0:len_data] = event['x'][0:len_data]
                input_array[1, 0:len_data] = event['y'][0:len_data]
                input_array[2, 0:len_data] = event['t'][0:len_data] / 1000 # to ms
            self.input.append(input_array)
            self.label.append(label)
    def __getitem__(self, i):
        # read data
        # image = cv2.imread(self.images_fps[i])
        
        # for k in event.keys():
        #     print(k, event[k])

        input_tensor= torch.Tensor(self.input[i])
        return input_tensor, torch.tensor(self.label[i])
    
    def __len__(self):
        return len(self.input)


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
        
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn5(self.fc2(x)))
        
        x = self.fc3(x)

        # iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        # if x.is_cuda:
        #     iden = iden.cuda()
        # x = x + iden
        # x = x.view(-1, 3, 3)
        return x

    
def format_logs(logs):
    str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
    s = ', '.join(str_logs)
    return s


def event_aug(input_array):
    choose = np.random.randint(1024, 2048)
    input_array[:,:,1024 : choose] = 0
    return input_array


def main():
    '''
    :return: None

    * :ref:`API in English <lif_fc_mnist.main-en>`

    .. _lif_fc_mnist.main-cn:

    使用全连接-LIF的网络结构，进行MNIST识别。\n
    这个函数会初始化网络进行训练，并显示训练过程中在测试集的正确率。

    * :ref:`中文API <lif_fc_mnist.main-cn>`

    .. _lif_fc_mnist.main-en:

    The network with FC-LIF structure for classifying MNIST.\n
    This function initials the network, starts trainingand shows accuracy on test dataset.
    '''
    args = parser.parse_args()
    root_dir = 'E:/BaiduNetdiskDownload/N-MNIST/'
   
    train_dataset = Dataset(root_dir, True)
    valid_dataset = Dataset(root_dir, False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=3, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)
    
    temp_data = train_dataset[4]
    print("########## Configurations ##########")
    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))
    print("####################################")
    # for x, y, x_len in train_data_loader:
    #     pass
    
    device = args.device
    dataset_dir = args.dataset_dir
    log_dir = args.log_dir
    model_output_dir = args.model_output_dir
    batch_size = args.batch_size 
    lr = args.lr
    T = args.T
    tau = args.tau
    train_epoch = args.epoch

    writer = SummaryWriter(log_dir)

    net = STN3d()
    # net(temp_data.unsqueeze(0))
    net = net.cuda()
    # 使用Adam优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # 使用泊松编码器
    encoder = encoding.PoissonEncoder()
    train_times = 0
    max_test_accuracy = 0

    test_accs = []
    train_accs = []
    loss_ce = torch.nn.CrossEntropyLoss()
    for epoch in range(train_epoch):
        print("Epoch {}:".format(epoch))
        print("Training...")
        train_correct_sum = 0
        train_sum = 0
        net.train()
        with tqdm(train_loader, file=sys.stdout) as iterator:
            for img, label in iterator:
                img = img.to(device)
                label = label.to(device)
                img = event_aug(img)
                # label_one_hot = F.one_hot(label, 10).float()
                # label_one_hot = F.one_hot(label, 10)
                # img = F.dropout(img, p = 0.2) * 0.8
                optimizer.zero_grad()
    
    
                output = net(img)
    
                # loss = F.mse_loss(output, label_one_hot)
                loss = loss_ce(output, label)
                loss.backward()
                optimizer.step()
                
    
                # 正确率的计算方法如下。认为输出层中脉冲发放频率最大的神经元的下标i是分类结果
                train_correct_sum += (output.max(1)[1] == label.to(device)).float().sum().item()
                train_sum += label.numel()
    
                train_batch_accuracy = (output.max(1)[1] == label.to(device)).float().mean().item()
                writer.add_scalar('train_batch_accuracy', train_batch_accuracy, train_times)
                train_accs.append(train_batch_accuracy)
    
                train_times += 1
                loss_logs = {'ce_loss': loss}
                iterator.set_postfix_str(format_logs(loss_logs))
        train_accuracy = train_correct_sum / train_sum

        print("Testing...")
        net.eval()
        with tqdm(valid_loader, file=sys.stdout) as iterator:
            # for img, label in iterator:
            with torch.no_grad():
                
                test_correct_sum = 0
                test_sum = 0
                for img, label in iterator:
                    img = img.to(device)
                    label = label.to(device)
                    label_one_hot = F.one_hot(label, 10)
                  
                    output = net(img)
                    loss = loss_ce(output, label)
                    loss_logs = {'ce_loss': loss}
                    
                    iterator.set_postfix_str(format_logs(loss_logs))
                    test_correct_sum += (output.max(1)[1] == label.to(device)).float().sum().item()
                    test_sum += label.numel()
                # functional.reset_net(net)
            test_accuracy = test_correct_sum / test_sum
            writer.add_scalar('test_accuracy', test_accuracy, epoch)
            test_accs.append(test_accuracy)
            max_test_accuracy = max(max_test_accuracy, test_accuracy)
        print("Epoch {}: train_acc = {}, test_acc={}, max_test_acc={}, train_times={}".format(epoch, train_accuracy, test_accuracy, max_test_accuracy, train_times))
        print()
    
    # 保存模型
    torch.save(net, model_output_dir + "/n_mnist.pt")
    # # 读取模型
    # # net = torch.load(model_output_dir + "/lif_snn_mnist.ckpt")

    # # 保存绘图用数据
    # net.eval()
    # # with torch.no_grad():
    # #     img, label = valid_dataset[0]        
    # #     img = img.to(device)
    # #     for t in range(T):
    # #         if t == 0:
    # #             out_spikes_counter = net(encoder(img).float())
    # #         else:
    # #             out_spikes_counter += net(encoder(img).float())
    # #     out_spikes_counter_frequency = (out_spikes_counter / T).cpu().numpy()
    # #     print(f'Firing rate: {out_spikes_counter_frequency}')
    # #     output_layer = net[-1] # 输出层
    # #     v_t_array = output_layer.v.cpu().numpy().squeeze().T  # v_t_array[i][j]表示神经元i在j时刻的电压值
    # #     np.save("v_t_array.npy",v_t_array)
    # #     s_t_array = output_layer.spike.cpu().numpy().squeeze().T  # s_t_array[i][j]表示神经元i在j时刻释放的脉冲，为0或1
    # #     np.save("s_t_array.npy",s_t_array)

    # train_accs = np.array(train_accs)
    # np.save('train_accs.npy', train_accs)
    # test_accs = np.array(test_accs)
    # np.save('test_accs.npy', test_accs)


if __name__ == '__main__':
    main()
