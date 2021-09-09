import config_utils

import logging
logging.basicConfig(format=config_utils.logging_format)
log = logging.getLogger('model_base')
log.setLevel(logging.DEBUG)

import os
import copy
import sys
import pdb
import shutil
import pickle
import json
import random

from collections import namedtuple, defaultdict, Counter
from pprint import pprint
from tqdm import tqdm

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import model_base

from functools import partial

import config_utils

def plot_results_XY(trainer, epoch):
    input_, target, output = trainer.eval_epoch(epoch)
    input_, target, output = [i.detach().cpu() for i in [input_, target, output]]
    plt.scatter(input_.cpu(), target.cpu(), label='x')
    plt.scatter(input_.cpu(), output.cpu(), label='x\'')
    plt.legend()
    plt.show()


def plot_results_TS(trainer, result):
    
    input_, target, output = result
    print('shapes: input_, target, ouptut: {}, {}, {}'.format(input_.size(),
                                                              target.size(),
                                                              output.size()))
    
    input_, target, output = [i.detach().cpu() for i in [input_, target, output]]
    plt.plot(range(target.size(0)), target.cpu(), label='x')
    plt.scatter(range(0, target.size(0)), output.cpu())
    #for i in range(target.size(1)): 
    #    plt.scatter(range(0, target.size(0)), output[:, i].cpu())
    
    plt.xlabel('time')
    plt.ylabel('population')
    plt.legend()
    plt.savefig(trainer.name + '_TS.png')
    plt.show()


def plot_ts_output(trainer, testsets, epoch=0):
    fig = plt.figure(figsize=(20, 15))
    for testset in testsets:
        input_, target, output = trainer.eval_epoch(-1, testset)
        print('shapes: input_, target, ouptut: {}, {}, {}'.format(input_.size(),
                                                                  target.size(),
                                                                  output.size()))
    
        input_, target, output = [i.detach().cpu() for i in [input_, target, output]]
        plt.scatter(range(0, target.size(0)), target.cpu(), label=testset.name)
        plt.scatter(range(0, target.size(0)), output.cpu(), label=testset.name)
        #for i in range(target.size(1)): 
        #    plt.scatter(range(0, target.size(0)), output[:, i].cpu())
        
    plt.xlabel('time')
    plt.ylabel('population')
    plt.legend()
    plt.savefig(trainer.config['hash'] + '/' + trainer.name + '_TS_{}.png'.format(epoch))
    plt.cla()    
    
def mse_loss(input, target):
    return torch.mean((input - target) ** 2)

def weighted_mse_loss(input, target, weight):
    return torch.mean(weight * (input - target) ** 2)


if __name__ == '__main__':

    config = json.load(open('../config.json'))
    hpconfig = json.load(open('hpconfig.json'))
    config['hpconfig_name'] = 'hpconfig'
    pprint(hpconfig)
    pprint(config)
    config_utils.init_config(config, hpconfig)
    assert config_utils.config != None

    dataset_path = config_utils.get_dataset_path_from_file(__file__)
    weights_path = config_utils.get_weights_path_from_file(__file__)
    model_name = os.path.splitext(os.path.basename(weights_path))[0]

    ts, vals = pickle.load(open(dataset_path, 'rb'))
    plt.plot(ts, vals)
    plt.show()
    
    ts = torch.Tensor(ts).view(-1 ,1)
    vals = torch.Tensor(vals)#.view(-1 ,1)
    print('ts: {}'.format(ts.size()))
    print('vals: {}'.format(vals.size()))
    
    if hpconfig['model'] == 'time-series':
        dlen = len(ts)
        pivot = int(0.5 * dlen)
        trainset = model_base.TSDataset(config_utils.config, hpconfig,
                                        'train', ts, vals)
        
        testset  = model_base.TSDataset(config_utils.config, hpconfig,
                                         'test', ts[pivot:], vals[pivot:])

        dataset = [trainset, testset]
        random_sample  = random.choice(testset)

        input_, output = random_sample
        
        print('random sample: ', random_sample)
        print('input_, output: {}, {}'.format(input_.size(), output.size()))

        model = model_base.TSModel(config_utils.config, hpconfig,
                                   input_.size()[-1],  output.size()[-1])
        
    if hpconfig['model'] == 'xy':     
        dataset = model_base.XYDataset(config_utils.config, hpconfig, ts, vals)

        random_sample  = random.choice(dataset)
        print('random sample: ', random_sample)
        input_, output = random_sample
        model = model_base.Model(config_utils.config, hpconfig,
                                 input_.size()[-1],  output.size()[-1])
        
    if os.path.exists(weights_path):
        print('loading old model....')
        model.load_state_dict(torch.load(weights_path))
    else:
        model.apply(model_base.weights_init_uniform)
        pass
    
    trainer = model_base.Trainer (
        config_utils.config,
        hpconfig,
        model_name,
        model,
        torch.nn.L1Loss(), #partial(weighted_mse_loss, weight=loss_weight.cuda() if config['cuda'] else loss_weight),
        torch.optim.Adam(model.parameters()),
        
        trainset,
        testset,        
        batch_size = 100,
        
        weights_path = weights_path
    )

    for i in range(100):
        trainer.do_train(1000)
        plot_ts_output(trainer, dataset, (i + 1) * 1000)

