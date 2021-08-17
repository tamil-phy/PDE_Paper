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

import CONFIG

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

    plt.plot(range(target.size(0)), target[:, 0].cpu(), '--', label='x')
    plt.scatter(range(target.size(0)), output[:, 0].cpu(), label='x\'')
    plt.plot(range(target.size(0)), target[:, 1].cpu(), '--', label='y')
    plt.scatter(range(target.size(0)), output[:, 1].cpu(), label='y\'')

    plt.xlabel('time')
    plt.ylabel('x and y')
    plt.legend()
    plt.savefig(trainer.name + '-tsplot.png')
    plt.show()
    
    plt.plot(target[:, 0], target[:, 1], label='x')
    plt.plot(output[:, 0], output[:, 1], '--', label='x\'')

    
    plt.plot(target[0, 0], target[0, 1], 'b<', label='x start', alpha=0.5, markersize=10)
    plt.plot(target[-1, 0], target[-1, 1], 'r<', label='x end', alpha=0.5, markersize=10)

    plt.plot(output[0, 0], output[0, 1], 'bo' , label='x\' start', alpha=0.5, markersize=10)
    plt.plot(output[-1, 0], output[-1, 1], 'ro', label='x\' end', alpha=0.5, markersize=10)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig(trainer.name + '-xyplot.png')
    plt.legend()
    plt.show()



if __name__ == '__main__':

    dataset_path = CONFIG.get_dataset_path_from_file(__file__)

    seq_length = 10
    resample_ratio = 1000
    ts, vals = pickle.load(open(dataset_path, 'rb'))
    dataset = model_base.TSDataset(ts[::resample_ratio], vals[::resample_ratio], seq_length)

    random_sample  = random.choice(dataset)
    print('random sample: ', random_sample)
    input_, output = random_sample
    model = model_base.TSModel(input_.size()[-1],
                               output.size()[-1],
                               100,
                               1,
                               seq_length)

   
    weights_path = CONFIG.get_weights_path_from_file(__file__)
    model_name = os.path.splitext(os.path.basename(weights_path))[0]
    if os.path.exists(weights_path):
        print('loading old model....')
        model.load_state_dict(torch.load(weights_path))
    else:
        model.apply(model_base.weights_init_uniform)
        pass
    
    trainer = model_base.Trainer (
        model_name,
        model,
        torch.nn.L1Loss(),
        torch.optim.Adam(model.parameters()),

        dataset,        
        dataset,
        batch_size = 10000,

        weights_path = weights_path
    )

    trainer.do_train(1000)

    plot_results_TS(trainer, -1)
