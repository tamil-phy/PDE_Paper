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

from torch import nn

from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt


import CONFIG

class HODataset(Dataset):
    def __init__(self, ts, vals):
        self.ts = torch.Tensor(ts).unsqueeze(1)
        self.vals = torch.Tensor(vals).unsqueeze(1)
        self.vals = self.vals / self.vals.abs().max()
        
        #pprint(list(zip(vals, self.vals.squeeze().tolist())))
        
    def __len__(self):
        return len(self.ts)

    def __getitem__(self, index):
        return [self.ts[index].squeeze().unsqueeze(0),
                self.vals[index].squeeze().unsqueeze(0)]
    
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
        self.model         = model         
        self.loss_function = loss_function  
        self.optimizer     = optimizer 
        
        
        self.trainset      = trainset      
        self.testset       = testset
        
        self.trainloader = DataLoader(trainset,
                                      #shuffle=True,
                                      batch_size = batch_size)

        self.testloader = DataLoader(testset,
                                     #shuffle=True,
                                     batch_size = batch_size)
        
        self.epochs       = epochs       
        self.every_nepoch = every_nepoch
        self.weights_path = weights_path 
        
        self.cuda = cuda
        if self.cuda:
            self.model.cuda()

    def validate_step(self, batch):
        image, label = batch
        if self.cuda:
            image, label = image.cuda(), label.cuda()
            
        output  =  self.model(image)
        loss    =  self.loss_function(output, label)
        accuracy = (output == label).float().mean()
        
        return loss, accuracy

    def eval_step(self, batch):
        image, label = batch
        if self.cuda:
            image, label = image.cuda(), label.cuda()
            
        output  =  self.model(image)
        return list(zip(image, label, output))

    def train_step(self, batch):
        self.optimizer.zero_grad()

        image, label = batch

        if self.cuda:
            image, label = image.cuda(), label.cuda()
            
        output  =  self.model(image)
        loss    =  self.loss_function(output, label)
        loss.backward()
        self.optimizer.step()

        return loss

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
    
    def do_train(self):
        train_loss = []
        epoch = 0
        loss = 1e10
        prev_loss = 1e10
        tbar = tqdm(range(self.epochs))
        save_count = 0
        for epoch in tbar:
            tbar.set_description('epoch:{} - loss:{:0.4f} - saves:{}'.format(epoch, loss, save_count))
            if epoch and epoch % self.every_nepoch == 0:
                loss, accuracy = self.validate_epoch(epoch)
                print('test epoch: {}, loss:{} accuracy: {}'.format(epoch, loss, accuracy))
                self.plot_results(epoch)     

            loss = self.train_epoch(epoch)
            #print("train epoch: {}, loss: {}".format(epoch, loss))

            if prev_loss > loss:
                prev_loss = loss
                if self.weights_path:
                    torch.save(copy.deepcopy(self.model).cpu().state_dict(), self.weights_path)
                    save_count += 1
                    
        return True
    
    def plot_results(self, epoch):
        input_, target, output = self.eval_epoch(epoch)
        input_, target, output = [i.detach().cpu() for i in [input_, target, output]]
        plt.scatter(input_.cpu(), target.cpu(), label='x')
        plt.scatter(input_.cpu(), output.cpu(), label='x\'')
        plt.legend()
        plt.show()



class Model(nn.Module):
    def __init__(self, input_size, output_size):

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

class Model(nn.Module):
    def __init__(self, input_size, output_size):

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

