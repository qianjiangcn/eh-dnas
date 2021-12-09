import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import yaml
import os
import sys
from hw_loss_model import LutLoss, EmbeddingFC
sys.path.insert(0, os.getcwd())


def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])


class Architect(object):

  def __init__(self, model, args):
    self.network_momentum = args.momentum
    self.network_weight_decay = args.weight_decay
    self.hw_loss_type = args.hw_loss_type
    self.hw_loss_rate = args.hw_loss_rate
    self.lut = self.load_yaml_content(args.lut)
    self.hw_loss =self.get_hw_loss()
    self.model = model
    self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
        lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

  def _compute_unrolled_model(self, input, target, eta, network_optimizer):
    loss = self.model._loss(input, target)
    theta = _concat(self.model.parameters()).data
    try:
      moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(self.network_momentum)
    except:
      moment = torch.zeros_like(theta)
    dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay*theta
    unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment+dtheta))
    return unrolled_model

  def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled):
    self.optimizer.zero_grad()
    if unrolled:
        self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
    else:
        self._backward_step(input_valid, target_valid)
    self.optimizer.step()

  def _backward_step(self, input_valid, target_valid):
    loss = self.model._loss(input_valid, target_valid)
    if self.hw_loss is not None:
      loss = loss + self.hw_loss_rate*self.hw_loss(self.model.arch_parameters())
    loss.backward()

  def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
    unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
    unrolled_loss = unrolled_model._loss(input_valid, target_valid)

    unrolled_loss.backward()
    dalpha = [v.grad for v in unrolled_model.arch_parameters()]
    vector = [v.grad.data for v in unrolled_model.parameters()]
    implicit_grads = self._hessian_vector_product(vector, input_train, target_train)

    for g, ig in zip(dalpha, implicit_grads):
      g.data.sub_(eta, ig.data)

    for v, g in zip(self.model.arch_parameters(), dalpha):
      if v.grad is None:
        v.grad = Variable(g.data)
      else:
        v.grad.data.copy_(g.data)
    if self.hw_loss is not None:
      hw_loss = self.hw_loss(unrolled_model.arch_parameters())
      hw_grad = torch.autograd.grad(hw_loss, unrolled_model.arch_parameters())
      for v,g in zip(self.model.arch_parameters(), hw_grad):
        v.data.add_(self.hw_loss_rate, g.data)

  def _construct_model_from_theta(self, theta):
    model_new = self.model.new()
    model_dict = self.model.state_dict()

    params, offset = {}, 0
    for k, v in self.model.named_parameters():
      v_length = np.prod(v.size())
      params[k] = theta[offset: offset+v_length].view(v.size())
      offset += v_length

    assert offset == len(theta)
    model_dict.update(params)
    model_new.load_state_dict(model_dict)
    return model_new.cuda()

  def _hessian_vector_product(self, vector, input, target, r=1e-2):
    R = r / _concat(vector).norm()
    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)
    loss = self.model._loss(input, target)
    grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.parameters(), vector):
      p.data.sub_(2*R, v)
    loss = self.model._loss(input, target)
    grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)

    return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

  def get_hw_loss(self):
    if self.hw_loss_type=='lut':
      hw_loss = LutLoss(self.lut)
      return hw_loss
    elif self.hw_loss_type=='pip':
      hw_emb = EmbeddingFC(28, 10).cuda()
      ckpt = torch.load(os.getcwd()+'/dnas/DARTS/'+'pip/model.pt')
      self.ckpt = ckpt
      self.hw_mean = 10691.551037565658
      self.hw_std = 2753.212163729927
      hw_emb.load_state_dict(ckpt)
      for param in hw_emb.parameters():
        param.requires_grad = False
      return hw_emb
    elif self.hw_loss_type=='acc':
      hw_emb = EmbeddingFC(28, 10).cuda()
      ckpt = torch.load(os.getcwd()+'/dnas/DARTS/'+'acc/model.pt')
      self.ckpt = ckpt
      self.hw_mean = 799.2808245914118
      self.hw_std = 147.87143187521818
      hw_emb.load_state_dict(ckpt)
      for param in hw_emb.parameters():
        param.requires_grad = False
      return hw_emb
    elif self.hw_loss_type==None:
      return None

  @staticmethod
  def load_yaml_content(yaml_path):
    with open(yaml_path, 'r') as f:
      d = yaml.safe_load(f)
    return d

