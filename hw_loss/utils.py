import sys
import os
import yaml
import torch
import numpy as np
import random
sys.path.insert(0, os.getcwd())

class ConfigLoader:
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
    def load_model_cfg(fname):
        cfg = ConfigLoader._load_yaml_content(fname)
        return cfg['model_params']

    @staticmethod
    def load_dataset_cfg(fname):
        cfg = ConfigLoader._load_yaml_content(fname)
        return cfg['dataset_params']

    @staticmethod
    def load_train_cfg(fname):
        cfg = ConfigLoader._load_yaml_content(fname)
        return cfg['train_params']

    @staticmethod
    def load_all_cfg(fname):
        cfg = ConfigLoader._load_yaml_content(fname)
        for key in cfg:
            print(key)
            print(cfg[key])
        return cfg['dataset_params'], cfg['model_params'], cfg['train_params']

def change_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


