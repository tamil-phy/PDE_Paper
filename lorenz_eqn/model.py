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
    
    input_, target, output = [i.detach().cpu()
                              for i in [input_, target, output]]
    
    fig = plt.figure(figsize=(12,10))
    # This is how we set the 3D plot
    ax = fig.gca(projection='3d')
    
    x,y,z = output[:, 0], output[:, 1], output[:, 2]
    ax.plot(x, y, z, lw=0.5,alpha=0.7)  # Plotting the values
    ax.scatter(x[-1], y[-1], z[-1], color=(1,0,0)) # Plotting the final values
    
    #Setting Axis Titles
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("Lorenz Attractor")
    plt.savefig(trainer.name + '.png')
    plt.show()



if __name__ == '__main__':

    filepath = 'data.pkl'
    seq_length = 19
    ts, vals = pickle.load(open(filepath, 'rb'))
    dataset = model_base.TSDataset(ts, vals, seq_length, merge_ts_vals=True)

    random_sample  = random.choice(dataset)
    print('random sample: ', random_sample)
    input_, output = random_sample
    model = model_base.TSModel(input_.size()[-1],
                               output.size()[-1],
                               100,
                               1,
                               seq_length,
                               )

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

    trainer.do_train(100)

    plot_results_TS(trainer, -1)
