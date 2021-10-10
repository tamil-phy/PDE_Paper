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

def plot_ts_output():
    fig, axs = plt.subplots(1, 6, figsize=(5, 4), dpi=600)

    paths = [
        ('(5h1L)', 'hpconfig__b7371c'),
        ('(20h1L)', 'hpconfig__e47951'),
        ('(25h1L)', 'hpconfig__797532'),
        ('(50h1L)', 'hpconfig__040b49'),
        ('(50h2L)', 'hpconfig__1f7ec8'),
    ]
    
    for i, (label, path) in enumerate(paths):
        path = '{}/{}'.format(path, 'korteweg_de_vries_TS_1000.pkl' )
        try:
            input_, target, output = pickle.load(open(path, 'rb'))
            
            axs[i].imshow(output, extent=[0, 50, 0, 200])
            axs[i].set_title(label)
            
            axs[-1].imshow(target, extent=[0, 50, 0, 200])
            axs[-1].set_title('target')

        except:
            print('error: ' + path)
            pass
        
    for i, (label, path) in enumerate(paths):
        path = '{}/{}'.format(path, 'korteweg_de_vries_TS_1000.pkl' )

        if i == 0 or i == len(paths):
            continue
        
        axs[i].set_yticks([])
        axs[i].set_yticklabels([])


        axs[-1].set_yticks([])
        axs[-1].set_yticklabels([])
        
    #axs[0].colorbar()
    plt.suptitle('Korteweg-de Vries\n(Periodic Domain)')

    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)
    
    plt.savefig('plot-compared.png')

    #plt.show()
    plt.cla()



plot_ts_output()
