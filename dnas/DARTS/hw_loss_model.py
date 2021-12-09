import torch
import torch.nn as nn
import yaml
import math
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Module
import collections

import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2

class LutLoss(nn.Module):
    def __init__(self, lut_dict, num_cells=8, edge_selection=False, reduction='mean'):
        super(LutLoss, self).__init__()
        self.lut_dict = lut_dict
        self.num_cells = num_cells
        self.edge_selection = edge_selection
        self.reduction = reduction
        self.lut = self.process_lut()

    def process_lut(self):
        op_list = ['op'+ str(i) for i in range(1,8)]
        cell_list = ['cell'+str(i) for i in range(self.num_cells)]
        lat_all = []
        for key in cell_list:
            cell = self.lut_dict[key]
            cell['op3'] = math.inf
            lat = [1/cell[k] for k in op_list]
            lat_all.append(lat)
        lat_tensor = torch.FloatTensor(lat_all)
        lat_mean_tensor = torch.mean(lat_tensor, dim=0)
        zero_op = torch.FloatTensor([0.0])
        full_lat = torch.cat([zero_op, lat_mean_tensor], dim=0).cuda()
        full_lat_2d = torch.unsqueeze(full_lat, dim=1)
        return Variable(full_lat_2d)

    def forward(self, als, edge_selection=False):
        if self.reduction=='mean':
            if self.edge_selection:
                als_normal = self.select_edge(als[0].data)
                als_reduce = self.select_edge(als[1].data)
                l1 = torch.sum(torch.matmul(als_normal, self.lut.data))
                l2 = torch.sum(torch.matmul(als_reduce, self.lut.data))
                return l1*6 + l2*2
            else:
                als_normal = F.softmax(als[0], dim=-1)
                als_reduce = F.softmax(als[1], dim=-1)
                l1 = torch.sum(torch.matmul(als_normal, self.lut))
                l2 = torch.sum(torch.matmul(als_reduce, self.lut))
            return l1+l2

    @staticmethod
    def select_edge(weights):
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
                _, idx = torch.max(weights[edge+start], dim=0)
                processed[edge+start, int(idx)] = 1.0
            start = end
            n += 1
        return processed


class EmbeddingFC(nn.Module):
    def __init__(self, emb_layer=28, emb_dim=10, dropout=0, edge_selection=False):
        super().__init__()
        self.dropout_rate = dropout
        self.emb_dim = emb_dim
        self.emb_layer = emb_layer
        emb_dict = collections.OrderedDict()
        for stage in range(self.emb_layer):
            emb_dict[str(stage)] = nn.Embedding(8, self.emb_dim)
        self.emb = nn.Sequential(emb_dict)
        self.fc1 = nn.Linear(self.emb_layer*self.emb_dim, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 1)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, als, edge_selection=False):

        if edge_selection:
            als_1 = Variable(self.select_edge(als[0].data))
            als_2 = Variable(self.select_edge(als[1].data))
            als = torch.cat([als_1, als_2], dim=0)
        else:
            als = torch.cat(als, dim=0)
        emb_list = []
        for stage in range(self.emb_layer):
            input_stage = als[stage:stage+1]
            emb_stage = torch.matmul(input_stage, Variable(self.emb[stage].weight.data))
            emb_list.append(emb_stage)
        emb = torch.stack(emb_list, dim=1)
        emb = emb.view(-1, self.emb_layer*self.emb_dim)
        out = F.relu(self.fc1(emb))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return 1/out

    @staticmethod
    def select_edge(weights):
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
                _, idx = torch.max(weights[edge+start], dim=0)
                processed[edge+start, int(idx)] = 1.0
            start = end
            n += 1
        return processed

