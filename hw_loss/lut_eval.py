import torch
import torch.utils.data as D
import sys
from tensorboardX import SummaryWriter
import os
sys.path.insert(0, os.getcwd())
import yaml
import  pickle

def lut_one_arch(lut, arch):
    arch = [int(i) for i in str(arch)]
    arch_normal = arch[:14]
    arch_reduce = arch[14:]
    l_normal = 0
    l_reduce = 0
    for cell in ['0','1','3','4','6','7']:
        lut_cell = lut['cell'+cell]
        l = 0
        for op in arch_normal:
            if op not in [0,3]:
                l = l + 1/lut_cell['op'+str(op)]
        l_normal += l
    for cell in ['2', '5']:
        edges_connect_to_input_node = [0,1,2,3,5,6,9,10]
        lut_cell = lut['cell'+cell]
        l = 0
        for edge, op in enumerate(arch_reduce):
            if edge in edges_connect_to_input_node:
                if op==3:
                    l = l + 1/lut_cell['facreduce']
                elif op!=0:
                    l = l + 1/lut_cell['op'+str(op)]
            elif op not in [0, 3]:
                l = l + 1/lut_cell['op'+str(op)]
        l_reduce += l
    l = l_normal + l_reduce
    return l


def main(path_to_lut, path_to_input, path_to_output):
    lut = load_yaml(path_to_lut)
    input = load_pickle(path_to_input)
    output = load_yaml(path_to_output)
    m_error = 0
    num = 200000
    for key in range(num):
        arch = input[str(key)]
        lut_est = lut_one_arch(lut, arch)
        thr_est = 1/lut_est
        gt = output[int(key)]
        error = abs(thr_est-gt)/gt
        m_error += error
    m_error = m_error/num
    print(m_error)

def load_yaml(path):
    with open(path,'r') as f:
        d = yaml.safe_load(f)
    return d
def load_pickle(path):
    with open(path, 'rb') as f:
        d = pickle.load(f)
    return d

if __name__ == '__main__':
    path_to_lut = 'our_nn/lut_overhead.yaml'
    path_to_input = 'model_zoo/DARTS/hwdataset_100w/100w_input_val.pkl'
    path_to_output = 'model_zoo/DARTS/hwdataset_100w/100w_output_val_acc.yaml'
    main(path_to_lut, path_to_input, path_to_output)
