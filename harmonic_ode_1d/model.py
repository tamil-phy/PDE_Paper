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

    plt.plot(target,  label='x')
    #plt.scatter(range(0, output.size(0), 1000), output[::1000], s=1, color='orange', label='x\'')
    plt.scatter(range(0, output.size(0)), output, s=10, color='orange', label='x\'')
    #plt.plot(output[-1, 0], output[-1], 'ro')
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('position')
    plt.grid()
    plt.savefig(trainer.name + '.png')
    plt.show()


if __name__ == '__main__':

    dataset_path = CONFIG.get_dataset_path_from_file(__file__)

    seq_length = 10
    ts, vals = pickle.load(open(dataset_path, 'rb'))
    dataset = model_base.TSDataset(ts[::1000], vals[::1000],  seq_length)

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
    print(weights_path)
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
        batch_size = 1000,

        weights_path = weights_path
    )

    trainer.do_train(100)
    plot_results_TS(trainer, -1)

