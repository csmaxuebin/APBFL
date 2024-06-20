import numpy as np
import copy
import datetime
import os
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset
import torch.nn.functional as F
# from src.noise import *

now = datetime.datetime.now()
timestamp = now.strftime("%Y%m%d_%H%M%S")
file_name = 'gradient_L2{}.txt'.format(timestamp)

# 指定文件路径和文件名
file_path = './results.txt/'

# 如果文件路径不存在，则创建它
if not os.path.exists(file_path):
    os.makedirs(file_path)

# 将文件路径和文件名合并起来
file_path_name = os.path.join(file_path, file_name)

class Client_ClusterFL(object):
    def __init__(self, name, model, local_bs, local_ep, lr, momentum, device, dp_clip, dp_epsilon, dp_delta, rounds,
                 train_dl_local=None, test_dl_local=None):

        self.name = name
        self.net = model
        self.local_bs = local_bs
        self.local_ep = local_ep
        self.lr = lr
        self.rounds = rounds
        self.momentum = momentum
        self.device = device
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = train_dl_local
        self.ldr_test = test_dl_local
        self.dp_clip = dp_clip
        self.dp_epsilon = dp_epsilon
        self.dp_delta = dp_delta
        self.acc_best = 0
        self.count = 0
        self.save_best = True

    def set_avg_epoch_grad_norm(self, norm_threshold_after):
        self.dp_clip = norm_threshold_after
        print("clip",self.dp_clip)


    def train(self,is_print=False):
        self.net.to(self.device)
        self.net.train()
        optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=0)
        epoch_loss = []
        for iteration in range(self.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.device), labels.to(self.device)
                self.net.zero_grad()
                log_probs = self.net(images)
                loss = self.loss_func(log_probs, labels.long())
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        # delta_net = copy.deepcopy(self.net)
        # # norm_before = self.compute_delta_net_norm(delta_net)
        # # delta_net = self.sparse_parameters_using_fisher(delta_net)
        # # norm_after = self.compute_delta_net_norm(delta_net)
        # with torch.no_grad():
        #     delta_net = self.clip_parameters(delta_net)
        # delta_net = self.add_noise(delta_net)
        #
        # self.net = copy.deepcopy(delta_net)
        return sum(epoch_loss) / len(epoch_loss)

    def get_state_dict(self):
        return self.net.state_dict()

    def get_best_acc(self):
        return self.acc_best

    def get_count(self):
        return self.count

    def get_net(self):
        return self.net

    def set_state_dict(self, state_dict):
        self.net.load_state_dict(state_dict)

    def eval_test(self):
        self.net.to(self.device)
        self.net.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.ldr_test:
                data, target = data.to(self.device), target.to(self.device)
                output = self.net(data)
                test_loss += F.cross_entropy(output, target.long(), reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        test_loss /= len(self.ldr_test.dataset)
        accuracy = 100. * correct / len(self.ldr_test.dataset)
        return test_loss, accuracy

    def eval_test_glob(self, glob_dl):
        # pacfl没用的这个函数
        self.net.to(self.device)
        self.net.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in glob_dl:
                data, target = data.to(self.device), target.to(self.device)
                output = self.net(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        test_loss /= len(glob_dl.dataset)
        accuracy = 100. * correct / len(glob_dl.dataset)
        return test_loss, accuracy

    def eval_train(self):
        # “eval_train”方法在不更新模型参数的情况下评估神经网络在训练数据上的性能
        self.net.to(self.device)
        self.net.eval()
        train_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.ldr_train:
                data, target = data.to(self.device), target.to(self.device)
                output = self.net(data)
                train_loss += F.cross_entropy(output, target.long(), reduction='sum').item()  # sum up batch loss
                pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
        train_loss /= len(self.ldr_train.dataset)
        accuracy = 100. * correct / len(self.ldr_train.dataset)
        return train_loss, accuracy