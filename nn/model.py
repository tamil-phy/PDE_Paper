import os
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
            batch_size = 10
    ):
        self.model = model
        self.loss_function = loss_function 
        self.optimizer     = optimizer      


        self.trainset = trainset
        self.testset = testset
        
        self.trainloader = DataLoader(trainset,
                                      #shuffle=True,
                                      batch_size = batch_size)

        self.testloader = DataLoader(testset,
                                     #shuffle=True,
                                     batch_size = batch_size)

        self.epochs = epochs
        self.cuda = cuda
        if self.cuda:
            self.model.cuda()

    def validate_step(self, batch):
        image, label = batch
        if self.cuda:
            image, label = image.cuda(), label.cuda()
            
        output  =  self.model(image)
        loss    =  self.loss_function(output, label)
        accuracy = (output.max(dim=1)[1] == label).float().mean()
        
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
            
        return torch.Tensor(outputs)
    
    def do_train(self):
        train_loss = []
        epoch = 0
        loss = 'computing...'
        for epoch in tqdm(range(self.epochs),
                          desc='epoch: {} - train loss: {}'.format(epoch, loss)):
            if epoch and epoch % 100 == 0:
                loss, accuracy = self.validate_epoch(epoch)
                print('test epoch: {}, loss:{} accuracy: {}'.format(epoch, loss, accuracy))
                
                self.plot_results(epoch)     

            loss = self.train_epoch(epoch)
            #print("train epoch: {}, loss: {}".format(epoch, loss))
            
        return True
    
    def plot_results(self, epoch):
        output_ = self.eval_epoch(epoch)
        input_, target, output = zip(*output_)

        plt.scatter(input_, target)
        plt.scatter(input_, output)
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
        
#         x = self.fc2(x)
#         x = torch.tanh(x)
        
#         x = self.fc3(x)
#         x = torch.tanh(x)
        
        x = self.fc4(x)
       
        return x



if __name__ == '__main__':

    filepath = 'harmonic_ode_1d.pkl'
    ts, vals = pickle.load(open('{}/{}'.format(CONFIG.DATA_DIR, filepath), 'rb'))
    dataset = HODataset(ts, -2 * ts + 4)
    dataset = HODataset(ts, vals)

    random_sample  = random.choice(dataset)
    print('random sample: ', random_sample)
    input_, output = random_sample
    model = Model(input_.size()[-1], output.size()[-1])
    #model.apply(weights_init_uniform)
    
    trainer = Trainer (
        model,
        torch.nn.L1Loss(),
        torch.optim.Adam(model.parameters()),

        dataset,        
        dataset,
        batch_size = 1000

    )

    trainer.do_train()
    plt.scatter(dataset.ts.cpu(), [model.forward(i.cuda()).cpu().detach() for  i in dataset.ts])
