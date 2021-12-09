import pickle
from time import time
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import yaml
import sys
import os
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


def get_ds(para, mode='darts'):
    if mode == 'darts':
        train_ds = DartsHwDataset('dnas/DARTS/hwdataset_100w/100w_input_train.pkl',
                                    'dnas/DARTS/hwdataset_100w/100w_output_train_'+para+'.pkl', para=para, prep=False)
        val_ds = DartsHwDataset('dnas/DARTS/hwdataset_100w/100w_input_val.pkl',
                                'dnas/DARTS/hwdataset_100w/100w_output_val_'+para+'.pkl', para=para, prep=False)
        test_ds = DartsHwDataset('dnas/DARTS/hwdataset_100w/100w_input_test.pkl',
                                    'dnas/DARTS/hwdataset_100w/100w_output_test_'+para+'.pkl', para=para, prep=False)
    return train_ds, val_ds, test_ds

class DartsHwDataset(Dataset):
    def __init__(self, input_file, output_file, para, prep=False):
        super(DartsHwDataset, self).__init__()
        self.para = para
        self.prep = prep
        self.output = self._load_pickle_content(output_file)
        self.input = self._load_pickle_content(input_file)
        self.content = self.match_dict_to_list()
        if self.para == 'pip':
            self.mean = 10691.551037565658
            self.std = 2753.212163729927
        elif self.para =='acc':
            self.mean = 799.2808245914118
            self.std = 147.87143187521818
    def __len__(self):
        return len(self.output)

    def __getitem__(self, idx):
        input, output = self.content[idx]
        input_list =  [int(i) for i in str(input)]
        als = torch.Tensor(input_list)
        output = (output - self.mean)/self.std
        return als, torch.Tensor([output])

    def match_dict_to_list(self):
        content = []
        for key in self.output:
            try:
                my_tuple = (self.input[str(key)], self.output[key])
                content.append(my_tuple)
            except KeyError:
                continue
        return content

    @staticmethod
    def preprocess(weights):
        processed = torch.zeros_like(weights)
        n = 2
        start = 0
        steps = 4
        for i in range(steps):
            end = start + n
            W = weights[start:end]
            edges = sorted(range(i+2), key=lambda x: -
                        max(W[x][k] for k in range(W[x].shape[0]) if k != 0))[:2]
            for edge in edges:
                idx = torch.argmax(weights[edge+start])
                processed[edge+start, idx] = 1.0
            start = end
            n += 1
        return processed


    @staticmethod
    def _load_yaml_content(fname):
        assert os.path.isfile(fname), 'Config file not found: {}'.format(fname)
        with open(fname, 'r') as stream:
            try:
                content = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
                sys.exit(-1)
        return content

    @staticmethod
    def _load_pickle_content(fname):
        with open(fname, 'rb') as f:
            dict1 = pickle.load(f)
        return dict1

def get_ds_stats(ds):
    outputs = [d[1] for d in ds]
    outputs = torch.cat(outputs)
    m = torch.mean(outputs)
    s = torch.std(outputs)
    return m, s
