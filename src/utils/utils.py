import torch
import random
import numpy as np
import datetime
import torch.nn as nn
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import torch.nn.functional as F
import torch.backends.cudnn
import yaml
import re
import csv
import math
import os, shutil

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def set_random_seed(seed, deterministic=True):
    """Set random seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def check_writable(path, overwrite=False):
    if not os.path.exists(path):
        os.makedirs(path)
    elif overwrite:
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        pass

def to_dense_batch(bg, feats,max_num_nodes=None):

    if max_num_nodes is None:
        max_num_nodes = int(bg.batch_num_nodes().max())
    batch = torch.cat([torch.full((1, x.type(torch.int)), y) for x, y in zip(bg.batch_num_nodes(), range(bg.batch_size))],
                    dim=1).reshape(-1).type(torch.long).to(bg.device)
    cum_nodes = torch.cat([batch.new_zeros(1), bg.batch_num_nodes().cumsum(dim=0)])
    idx = torch.arange(bg.num_nodes(), dtype=torch.long, device=bg.device)
    idx = (idx - cum_nodes[batch]) + (batch * max_num_nodes)
    size = [bg.batch_size * max_num_nodes] + list(feats.size())[1:]
    out = feats.new_full(size, fill_value=0)
    out[idx] = feats
    out = out.view([bg.batch_size, max_num_nodes] + list(feats.size())[1:])
    mask = torch.zeros(bg.batch_size, max_num_nodes, dtype=torch.bool, device=bg.device)
    batch_num_nodes = bg.batch_num_nodes()
    for i in range(bg.batch_size):
        mask[i, :batch_num_nodes[i]] = True

    return out, mask
    

class EarlyStopping(object):
    def __init__(self,  mode='higher', patience=15, filename=None, tolerance=0.0):
        if filename is None:
            dt = datetime.datetime.now()
            filename = './save/early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(dt.date(), dt.hour, dt.minute, dt.second)

        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower

        self.patience = patience
        self.tolerance = tolerance
        self.counter = 0
        self.filename = filename
        self.best_score = None
        self.early_stop = False

    def _check_higher(self, score, prev_best_score):
        # return (score > prev_best_score)
        return score / prev_best_score > 1 + self.tolerance

    def _check_lower(self, score, prev_best_score):
        # return (score < prev_best_score)
        return prev_best_score / score > 1 + self.tolerance

    def step(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience} best val: {self.best_score :.4f}')
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, model):
        '''Saves model when the metric on the validation set gets improved.'''
        torch.save({'model_state_dict': model.state_dict()}, self.filename)  

    def load_checkpoint(self, model):
        '''Load model saved with early stopping.'''
        model.load_state_dict(torch.load(self.filename)['model_state_dict'])







