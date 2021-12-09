import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
import pdb


def get_model(model_cfg):
    input_size = model_cfg['num_layer']*model_cfg['num_op_per_layer']
    if model_cfg['model'] == 'linear1':
        model = Linear1(input_size)
    elif model_cfg['model'] == 'linear2':
        model = Linear2(input_size)
    elif model_cfg['model'] == 'linear3':
        model = Linear3(input_size)
    elif model_cfg['model'] == 'embed':
        model = EmbeddingFC(model_cfg['num_layer'],
                            model_cfg['emb_dim'],model_cfg['dropout'])
    else:
        raise Exception('Model not defined')
    return model.cuda()

class Linear1(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(self.input_size, 1)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        return self.fc1(x)

class Linear2(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(self.input_size, 20)
        self.fc2 = nn.Linear(20, 1)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Linear3(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(self.input_size, 100)
        self.fc2 = nn.Linear(100, 20)
        self.fc3 = nn.Linear(20, 1)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class EmbeddingFC(nn.Module):
    def __init__(self, emb_layer, emb_dim, dropout):
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

    def forward(self, x):
        emb_list = []
        for stage in range(self.emb_layer):
            input_stage = x[:,stage].long()
            emb_stage = self.emb[stage](input_stage)
            emb_list.append(emb_stage)
        emb = torch.stack(emb_list, dim=1)
        emb = self.dropout(emb)
        emb = emb.view(-1, self.emb_layer*self.emb_dim)
        out = F.relu(self.fc1(emb))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
