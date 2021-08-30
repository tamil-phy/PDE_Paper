import aconfig

import logging
logging.basicConfig(format=aconfig.logging_format)
log = logging.getLogger('model_base')
log.setLevel(logging.DEBUG)

import os
import copy
import sys
import pdb
import shutil
import pickle
import json
from collections import namedtuple, defaultdict, Counter
from pprint import pprint
from tqdm import tqdm

import random
import torch

import numpy as np
from torch import nn

from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt

class XYDataset(Dataset):
    def __init__(self, config, hpconfig,  X, Y):
        assert len(X) == len(Y), '{} != {}'.format(len(X), len(Y))
        resample_ratio = len(X) // hpconfig['resample_ratio']
        X = X[::resample_ratio]
        Y = Y[::resample_ratio]

        self.X = torch.Tensor(X)
        self.Y = torch.Tensor(Y)
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return [self.X[index],
                self.Y[index]]


class TSDataset(Dataset):
    def __init__(self, config, hpconfig, input_, output):
        
        def sliding_windows(input_, output, seq_length):
            X = []
            Y = []
            
            for i in range( len(input_) - seq_length - 1 ):
                x = input_[i:(i+seq_length)]
                y = output[i+seq_length]
                X.append(x)
                Y.append([y])

            return torch.Tensor(X), torch.Tensor(Y)

        self.config   = config
        self.hpconfig = hpconfig
        assert len(input_) == len(output) ,\
            '{} != {}'.format(len(input_), len(output))
        
        input_, output = np.array(input_), np.array(output)
        resample_ratio = len(input_) // hpconfig['resample_ratio']
        input_ = input_[::resample_ratio]
        output    = output   [::resample_ratio]
        
        self.input_, self.output = sliding_windows(input_, output, hpconfig['seq_length'])
        print('shapes: input_, output: {}, {}'.format(self.input_.size(), self.output.size()))

        plt.plot(self.output)
        plt.show()
        
    def __len__(self):
        return len(self.input_)

    def __getitem__(self, index):
        return [self.input_[index],
                self.output[index]]

    def __add__(self, other):
        new = copy.deepcopy(other)
        new.input_ = torch.Tensor(self.input_.tolist() + other.input_.tolist())
        new.output = torch.Tensor(self.output.tolist() + other.output.tolist())

        return new
    
# takes in a module and applies the specified weight initialization
def weights_init_uniform(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # apply a uniform distribution to the weights and a bias=0
        m.weight.data.uniform_(0.0, 1.0)
        m.bias.data.fill_(0)

        
class Trainer:
    def __init__(
            self,
            config,
            hpconfig,
            name,
            model,
            loss_function,
            optimizer,

            trainset,
            testset,
            cuda = True,
            epochs = 1000000,
            every_nepoch = 1000,
            batch_size = 10,
            weights_path = None
    ):

        self.config = config
        self.hpconfig = hpconfig
        self.name = name
        
        self.model         = model         
        self.loss_function = loss_function  
        self.optimizer     = optimizer 
        
        
        self.trainset      = trainset      
        self.testset       = testset
        
        self.trainloader = DataLoader(trainset,
                                      shuffle=False,
                                      batch_size = batch_size)

        self.testloader = DataLoader(testset,
                                     shuffle=False,
                                     batch_size = batch_size)
        
        self.epochs       = epochs       
        self.every_nepoch = every_nepoch
        self.weights_path = weights_path 

        self.loss_records     = []
        self.accuracy_records = []
        
        self.cuda = config['cuda']
        self.cuda = cuda #function args overrides config.cuda
        if self.cuda:
            self.model.cuda()


    def write_metric(self, metric, path):
        with open(self.config['metrics_path'][path], 'w') as f:
            for rec in self.loss_records:
                f.write('{}\t{}'.format(*rec))

    
    # step functions for train, test, eval
    # handles one batch at a time
    # they all have same input signature
    def validate_step(self, batch):
        input_, target = batch
        if self.cuda:
            input_, target = input_.cuda(), target.cuda()
            
        output  =  self.model(input_)
        loss    =  self.loss_function(output, target)
        accuracy = (output == output).float().mean()
        
        return loss, accuracy

    def eval_step(self, batch):
        input_, target = batch
        if self.cuda:
            input_, target = input_.cuda(), target.cuda()
            
        output  =  self.model(input_)
        return list(zip(input_, target, output))

    def train_step(self, batch):
        self.optimizer.zero_grad()

        input_, target = batch

        if self.cuda:
            input_, target = input_.cuda(), target.cuda()
            
        output  =  self.model(input_)
        loss    =  self.loss_function(output, target)
        loss.backward()
        self.optimizer.step()

        return loss

    # epoch level functions 
    def train_epoch(self, epoch):
        self.model.train()
        losses = []
        
        for batch in self.trainloader:
            loss= self.train_step(batch)
            
            losses.append(loss)
            
        return torch.stack(losses).mean().item()
    
    def validate_epoch(self, epoch):
        self.model.eval()
        losses     = []
        accuracies = []
        for batch in self.testloader:
            loss, accuracy = self.validate_step(batch)
            
            losses.append(loss)
            accuracies.append(accuracy)
            
        return torch.stack(losses).mean().item(), torch.stack(accuracies).mean().item()
    
    def eval_epoch(self, epoch):
        self.model.eval()
        outputs     = []
        for batch in self.testloader:
            output = self.eval_step(batch)
            outputs.extend(output)

        return [torch.stack(i).squeeze() for i in  zip(*outputs)]

    # the actual training loop
    def do_train(self, epochs=0):
        train_loss = []
        epoch = 0
        loss = 1e10
        prev_loss = 1e10
        epoch_bar = tqdm(range(epochs or self.epochs))
        save_count = 0
        for epoch in epoch_bar:
            epoch_bar.set_description('epoch:{} - loss:{:0.4f} - saves:{}'.format(epoch, loss, save_count))
            if epoch and epoch % self.every_nepoch == 0:
                loss, accuracy = self.validate_epoch(epoch)
                self.accuracy_records.append((epoch, accuracy))

                print('test epoch: {}, loss:{} accuracy: {}'.format(epoch, loss, accuracy))
                self.write_metric(loss, 'loss_path')
                self.write_metric(accuracy, 'accuracy_path')
                
            loss = self.train_epoch(epoch)
            self.loss_records.append((epoch, loss))

            if prev_loss > loss:
                prev_loss = loss
                if self.weights_path:
                    torch.save(copy.deepcopy(self.model).cpu().state_dict(), self.weights_path)
                    save_count += 1
                    
        return True
    



class Model(nn.Module):
    def __init__(self, config, hpconfig, input_size, output_size):

        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        #self.fc2 = nn.Linear(64, 64)
        #self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)
        
        #x = self.fc2(x)
        #x = torch.tanh(x)
         
        #x = self.fc3(x)
        #x = torch.tanh(x)
        
        x = self.fc4(x)
       
        return x


class TSModel(nn.Module):

    def __init__(self, config, hpconfig, input_size, output_size):
        super().__init__()

        self.hpconfig = hpconfig
        
        self.input_size = input_size
        self.output_size = output_size

        self.num_layers = hpconfig['num_layers']
        self.hidden_size = hpconfig['hidden_size']
        self.seq_length  = hpconfig['seq_length']
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True)
        
        self.fc = nn.Linear(self.hidden_size * self.num_layers, output_size)

    def forward(self, x):
        h_0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_size)
        
        c_0 = torch.zeros(
            self.num_layers, x.size(0), self.hidden_size)

        h_0, c_0 = h_0.to(x.device), c_0.to(x.device)
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        
        h_out = h_out.view(-1, self.num_layers * self.hidden_size)
        
        out = self.fc(torch.relu(h_out))
        
        return out
