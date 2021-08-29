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

import config_utils

def plot_results_XY(trainer, epoch):
    input_, target, output = trainer.eval_epoch(epoch)
    input_, target, output = [i.detach().cpu() for i in [input_, target, output]]
    plt.scatter(input_.cpu(), target.cpu(), label='x')
    plt.scatter(input_.cpu(), output.cpu(), label='x\'')
    plt.legend()
    plt.show()


def plot_results_TS(trainer, epoch):
    input_, target, output = trainer.eval_epoch(epoch)
    print('shapes: input_, target, ouptut: {}, {}, {}'.format(input_.size(),
                                                              target.size(),
                                                              output.size()))
    
    input_, target, output = [i.detach().cpu() for i in [input_, target, output]]
    plt.plot(range(target.size(0)), target.cpu(), label='x')
    plt.scatter(range(0, target.size(0), 100), output[::100].cpu(), label='x\'', color='orange')
    
    #plt.plot(range(target.size(0)), target[:, 1].cpu(), label='x')
    #plt.scatter(range(target.size(0)), output[:, 1].cpu(), label='x\'')
    #plt.plot(range(target.size(0)), target[:, 2].cpu(), label='y')
    #plt.scatter(range(target.size(0)), output[:, 2].cpu(), label='y\'')
    #plt.plot(range(target.size(0)), target[:, 3].cpu(), label='z')
    #plt.scatter(range(target.size(0)), output[:, 3].cpu(), label='z\'')

    plt.xlabel('time')
    plt.ylabel('population')
    plt.legend()
    plt.savefig(trainer.name + '.png')
    plt.show()
    

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

    ts, vals, K = pickle.load(open(dataset_path, 'rb'))
    
    ts = torch.Tensor(ts)
    vals = torch.Tensor(vals)
    print('ts: {}'.format(ts.size()))
    print('vals: {}'.format(vals.size()))
    kth_samples = {}
    for ki, k in enumerate(K):
        k = torch.Tensor([k]).expand_as(ts)
        log.debug('sizes: k, v: {}, {}'.format(k.size(), vals[:, ki].size()))
        kth_samples[k] = torch.cat([vals[:, ki].unsqueeze(1), k])
        
    if hpconfig['model'] == 'time-series':
        dataset = []
        for k, samples in kth_samples.items():
            dataset.append(model_base.TSDataset(config_utils.config, hpconfig, ts, vals))

        d = dataset[0] 
        for di in dataset[1:]:
            d = d + di

        dataset = d
        random_sample  = random.choice(dataset)
        print('random sample: ', random_sample)
        input_, output = random_sample
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
        torch.nn.L1Loss(),
        torch.optim.Adam(model.parameters()),
        
        dataset,        
        dataset,
        batch_size = 100,
        
        weights_path = weights_path
    )
    
    trainer.do_train(1000)
