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
import wandb
import os
import sys

parser = argparse.ArgumentParser(description='spikingjelly LIF MNIST Training')

parser.add_argument('--device', default='cuda:0', help='运行的设备，例如“cpu”或“cuda:0”\n Device, e.g., "cpu" or "cuda:0"')

parser.add_argument('--dataset_dir', default='./', help='保存MNIST数据集的位置，例如“./”\n Root directory for saving MNIST dataset, e.g., "./"')
parser.add_argument('--log_dir', default='./result', help='the path of sving result')
parser.add_argument('--model-output-dir', default='./result', help='模型保存路径，例如“./”\n Model directory for saving, e.g., "./"')

parser.add_argument('--load_dir', default='./result/exp1/n_mnist.pt', help='load model path')

parser.add_argument('-b', '--batch-size', default=128, type=int, help='Batch 大小，例如“64”\n Batch size, e.g., "64"')
parser.add_argument('-T', '--timesteps', default=100, type=int, dest='T', help='仿真时长，例如“100”\n Simulating timesteps, e.g., "100"')
parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float, metavar='LR', help='学习率，例如“1e-3”\n Learning rate, e.g., "1e-3": ', dest='lr')
# parser.add_argument('--tau', default=2.0, type=float, help='LIF神经元的时间常数tau，例如“100.0”\n Membrane time constant, tau, for LIF neurons, e.g., "100.0"')
parser.add_argument('-N', '--epoch', default=300, type=int, help='训练epoch，例如“100”\n Training epoch, e.g., "100"')



from torch.autograd import Variable


def event_aug(input_array):
    if np.random.randn() < 0.5:
        sigma = 1
        input_array += torch.abs(sigma * torch.randn(input_array.shape).int())
    idx = np.arange(input_array.shape[1])
    
    # if np.random.randn() < 0.5:        
    np.random.shuffle(idx)
    
    # if np.random.randn() < 0.5:     
    #     x_pad = np.random.randint(-4, 4)
    #     y_pad = np.random.randint(-4, 4)
    #     input_array[0,:] += x_pad
    #     input_array[1,:] += y_pad
    #     input_array[0,:][input_array[0,:] < 0] = np.random.randint(0, 34)
        
    #     input_array[1,:][input_array[1,:] < 0] = np.random.randint(0, 34)
        
    #     input_array[0,:][input_array[0,:] > 33] = np.random.randint(0, 34)
        
    #     input_array[1,:][input_array[1,:] > 33] = np.random.randint(0, 34)
    
    # if np.random.randn() < 0.5:        
    #     input_array[0,:] = input_array[0,:].max() - input_array[0,:]
    # if np.random.randn() < 0.5:        
    #     input_array[:,1,:] = input_array[:,1,:].max() - input_array[:,1,:]
    # if np.random.randn(1) < 0.5:
    #     input_array[:,0,:] = 27 - input_array[:,0,:]
        # input_array[:,1,:]
    # choose = np.random.randint(1024, 2048)
    # input_array[:,:,1024 : choose] = 0
    input_array[0,:] / 34  # normallize ratio
    input_array[1,:] / 34  # normallize ratio
    input_array[2,:] / 316  # normallize ratio
    return input_array[:,idx]

class Dataset(BaseDataset):
    """
    For time-base evnet data
    
    """
    
    def __init__(
        self, 
        images_dir,
        train_mode,
        aug = True,
        cache = True,
        # event_set,
    ):
        # ids = os.listdir(images_dir)
        self.images_dir = images_dir
        self.event_set = NMNIST(images_dir, train=train_mode, data_type='event')
         # event, label = event_set[0]
        self.input = []
        self.label = []
        self.aug = aug
        self.cache = cache
        if self.cache:
            for i in tqdm(range(len(self.event_set))):
            # for i in tqdm(range(500)):            
                event, label = self.event_set[i]
                len_data = len(event['t'])
                max_index = np.nonzero(event['t'][0:] <= 100000)[0][-1]
                input_array = np.zeros([3, max_index], np.float)
                # if max_index < 1536:
                input_array[0, 0:max_index] = event['x'][0:max_index]
                input_array[1, 0:max_index] = event['y'][0:max_index]
                input_array[2, 0:max_index] = event['t'][0:max_index] / 1000 # to ms
                    # index = np.random.choice(np.arange(max_index), size=, replace=False)
                # else:
                    
                # if len_data > 1536:
                #     input_array[0, 0:len_data] = event['x'][0:1536]
                #     input_array[1, 0:len_data] = event['y'][0:1536]
                #     input_array[2, 0:len_data] = event['t'][0:1536] / 1000 # to ms
                # #     if event['t'][0:2048].max() > max_size:
                # #         max_size = event['t'][0:2048].max()
                # #         print(event['t'][0:2048].max())
                # # # input_array
                # else:
                #     input_array[0, 0:len_data] = event['x'][0:len_data]
                #     input_array[1, 0:len_data] = event['y'][0:len_data]
                #     input_array[2, 0:len_data] = event['t'][0:len_data] / 1000 # to ms
                #     # if event['t'][0:len_data].max() > max_size:
                #     #     max_size = event['t'][0:len_data].max()
                #     #     print(event['t'][0:len_data].max())
                self.input.append(input_array)
                self.label.append(label)
    
    
    def down_sample(self, input_array, sample_num):
        # input_array = np.zeros([3, 1536], np.float)
        
        index = np.random.choice(np.arange(len(input_array[0])), size=sample_num, replace=False)
        return input_array[:,index]
    
    
    def up_sample(self, input_array, sample_num):
        temp_array = np.zeros([3, sample_num], np.float)
        # print(len(input_array))
        # polar = len(input_array[0])
        
        concat_array = input_array
        while len(concat_array[0]) < sample_num:
            concat_array = np.concatenate((concat_array, input_array), axis=1)  
        temp_array = self.down_sample(concat_array, sample_num)
        # index = np.random.choice(np.arange(len(concat_array[0])), size=sample_num - len(concat_array[0]), replace=False)
        
        # temp_array[:, 0:len(concat_array[0])] = concat_array
        # temp_array[:, len(concat_array[0]):] = concat_array[:, index]
        
        # while polar < sample_num:
        #     if sample_num - polar > polar:
        #         # pass
        #         # index = np.random.choice(np.arange(len(input_array[0])), size=len(input_array[0]), replace=False)
        #         temp_array[:, 0:polar] = input_array
        #         temp_array[:, polar: polar + len(input_array[0])] = input_array
        #         polar += len(input_array[0])
        #     else:
        #         index = np.random.choice(np.arange(len(input_array[0])), size=sample_num - len(input_array[0]), replace=False)
        #         temp_array[:, 0:len(input_array[0])] = input_array
        #         temp_array[:, len(input_array[0]):] = input_array[:, index]
        #         break
        return temp_array
    
    
    def __getitem__(self, i):
        # read data
        # image = cv2.imread(self.images_fps[i])
        
        # for k in event.keys():
        #     print(k, event[k])

        if self.cache:
            input_array = self.input[i]
            # print(np.nonzero(input_array[0])[0][-1])
            if np.nonzero(input_array[0])[0][-1] > 1536:
                input_array = self.down_sample(input_array, 1536)
            else:
                input_array = self.up_sample(input_array, 1536)
            
            input_tensor= torch.Tensor(input_array)
            label = torch.tensor(self.label[i])
            
        else:
            event, label = self.event_set[i]
            len_data = len(event['t'])
            max_index = np.nonzero(event['t'][0:] <= 100000)[0][-1]
            input_array = np.zeros([3, 4096], np.float)
            # if max_index < 1536:
            input_array[0, 0:max_index] = event['x'][0:max_index]
            input_array[1, 0:max_index] = event['y'][0:max_index]
            input_array[2, 0:max_index] = event['t'][0:max_index] / 1000 # to 
            if max_index >= 1536:
                input_array = self.down_sample(input_array, 1536)
            else:
                input_array = self.up_sample(input_array, 1536)
                
            input_tensor= torch.Tensor(input_array)
            label = torch.tensor(self.label[i])
        if self.aug:
            input_tensor = event_aug(input_tensor)
        return input_tensor, label
    
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
        
        self.dropout1 = nn.Dropout(p=0.4)
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
        # x = self.dropout1(x)
        x = F.relu(self.bn5(self.dropout1(self.fc2(x))))
        
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


def main():
    '''
    :return: None
   
    '''
    args = parser.parse_args()
    root_dir = 'E:/BaiduNetdiskDownload/N-MNIST/'
    save_dir = args.log_dir
    experiment = wandb.init(project='cloud_event_net', resume='allow', anonymous='must')
    
    
    list_dirs = os.listdir(save_dir)
    result_dir = os.path.join(save_dir, 'exp' + str(len(list_dirs)))
    os.mkdir(result_dir)
    train_dataset = Dataset(root_dir, True)
    valid_dataset = Dataset(root_dir, False)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)
    
    temp_data = train_dataset[100]
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
    # tau = args.tau
    train_epoch = args.epoch
    experiment.config.update(dict(epochs=train_epoch, batch_size=args.batch_size, learning_rate=args.lr))
    writer = SummaryWriter(log_dir)

    net = STN3d()
    # net(temp_data.unsqueeze(0))
    net = net.cuda()
    # 使用Adam优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=train_epoch)
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
                # img = event_aug(img)
                # label_one_hot = F.one_hot(label, 10).float()
                # label_one_hot = F.one_hot(label, 10)
                # img = F.dropout(img, p = 0.2) * 0.8
                optimizer.zero_grad()
    
                
                output = net(img)
    
                # loss = F.mse_loss(output, label_one_hot)
                loss = loss_ce(output, label)
                loss.backward()
                optimizer.step()
                scheduler.step()
    
                train_correct_sum += (output.max(1)[1] == label.to(device)).float().sum().item()
                train_sum += label.numel()
    
                train_batch_accuracy = (output.max(1)[1] == label.to(device)).float().mean().item()
                writer.add_scalar('train_batch_accuracy', train_batch_accuracy, train_times)
                train_accs.append(train_batch_accuracy)
    
                train_times += 1
                loss_logs = {'ce_loss': loss}
                iterator.set_postfix_str(format_logs(loss_logs))
        train_accuracy = train_correct_sum / train_sum
        experiment.log({
            'train_accuracy': train_accuracy,
            'epoch': epoch
        })
        print("Testing...")
        net.eval()
        total_loss = 0
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
                    
                    total_loss += F.cross_entropy(output, label, reduction='sum').item() 
                    loss_logs = {'ce_loss': loss}
                    
                    iterator.set_postfix_str(format_logs(loss_logs))
                    test_correct_sum += (output.max(1)[1] == label.to(device)).float().sum().item()
                    test_sum += label.numel()
            total_loss /= len(valid_loader.dataset)
                # functional.reset_net(net)
            test_accuracy = test_correct_sum / test_sum
            writer.add_scalar('test_accuracy', test_accuracy, epoch)
            test_accs.append(test_accuracy)
            if test_accuracy >= max_test_accuracy:
                torch.save(net.state_dict(), os.path.join(result_dir, str(round(test_accuracy, 5)) + '_best.pth'))
                print('Model saved!')
            max_test_accuracy = max(max_test_accuracy, test_accuracy)
        experiment.log({
            'test_accuracy': test_accuracy,
            'total_loss': total_loss,
            'epoch': epoch
        })
        print("Epoch {}: train_acc = {}, test_acc={}, max_test_acc={}, train_times={}".format(epoch, train_accuracy, test_accuracy, max_test_accuracy, train_times))
        # print()
    
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
