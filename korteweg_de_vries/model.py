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

    

    fig, axs = plt.subplots(2, 1, figsize=(2, 10))
    
    axs[0].imshow(output, extent=[0, 50, 0, 200])
    axs[0].set_title('output')
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('t')
    
    axs[1].imshow(target, extent=[0, 50, 0, 200])
    axs[1].set_title('target')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('t')

    #axs[0].colorbar()
    plt.suptitle('Korteweg-de Vries\n(Periodic Domain)')
    plt.savefig(trainer.name + '.png')
    plt.show()



if __name__ == '__main__':

    dataset_path = CONFIG.get_dataset_path_from_file(__file__)

    seq_length = 10
    ts, vals = pickle.load(open(dataset_path, 'rb'))
    dataset = model_base.TSDataset(ts, vals, seq_length)

    random_sample  = random.choice(dataset)
    print('random sample: ', random_sample)
    input_, output = random_sample
    model = model_base.TSModel(input_.size()[-1],
                               output.size()[-1],
                               50,
                               1,
                               seq_length)

    weights_path = CONFIG.get_weights_path_from_file(__file__)
    model_name = os.path.splitext(os.path.basename(weights_path))[0]
      
    if os.path.exists(weights_path):
        try:
            print('loading old model....: {}'.format(weights_path))
            model.load_state_dict(torch.load(weights_path))
        except:
            print('loading model failed')
            model.apply(model_base.weights_init_uniform)
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
        epochs = 10000,
        weights_path = weights_path
    )

    plot_results_TS(trainer, 0)
    
    trainer.do_train()

    plot_results_TS(trainer, 0)
