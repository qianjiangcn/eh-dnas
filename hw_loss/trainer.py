from os import device_encoding
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.autograd import Variable

def train(model, writer, run, train_cfg, train_loader, val_loader):
    best_loss = float('inf')
    best_err = float('inf')
    if train_cfg['optimizer']=='adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr=train_cfg['lr'], weight_decay=train_cfg['wd'])
    elif train_cfg['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), lr=train_cfg['lr'], momentum=0.9, weight_decay=train_cfg['wd'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[100, 200, 300, 400], gamma=0.5)
    for epoch in range(train_cfg['epochs']):
        if epoch%1 == 0:
            val_loss, val_err = eval(model, train_cfg, val_loader)
            print('Epoch{} val loss {}, error rate {}'.format(epoch, val_loss, val_err))
            writer.add_scalar(str(run)+'/val_loss', val_loss, epoch)
            writer.add_scalar(str(run)+'/val_err', val_err, epoch)
            if val_err < best_err:
                best_err = val_err
                best_epoch = epoch
                print('Best err {} at epoch{}'.format(val_err, epoch))
                try:
                    os.mkdir(train_cfg['dir_ckpts']+train_cfg['exp_name'])
                except FileExistsError:
                    pass
                torch.save(model.state_dict(), train_cfg['dir_ckpts']+train_cfg['exp_name']+"/model.pt")
        model.train()
        ep_loss = 0
        for input, output in train_loader:
            optimizer.zero_grad()
            input = Variable(input).cuda()
            output = Variable(output).cuda()
            pred = model(input)
            if train_cfg['criterion'] == 'l1':
                loss = F.l1_loss(pred, output, reduction='sum')
            elif train_cfg['criterion'] == 'l2':
                loss = F.mse_loss(pred, output, reduction='sum')
            ep_loss += loss.item()
            loss.backward()
            optimizer.step()
        scheduler.step()
        print('Epoch{} train loss :{}'.format(epoch, ep_loss/len(train_loader.dataset)))
        writer.add_scalar(str(run)+'/train_loss', ep_loss /
                          len(train_loader.dataset), epoch)
        writer.add_scalar(
            str(run)+'/lr', optimizer.param_groups[0]['lr'], epoch)



def eval(model, train_cfg, loader):
    model.eval()
    loss = 0
    error = 0
    for input, output in loader:
        input = Variable(input).cuda()
        output = Variable(output).cuda()
        pred = model(input)
        if train_cfg['criterion'] == 'l1':
            loss_bs = F.l1_loss(pred, output, reduction='sum')
        elif train_cfg['criterion'] == 'l2':
            loss_bs = F.mse_loss(pred, output, reduction='sum')
        loss += loss_bs.item()
        mean = loader.dataset.mean
        std = loader.dataset.std
        pred = pred*std + mean
        output = output*std + mean
        error += torch.sum(torch.div(torch.abs(pred - output), torch.abs(output))).item()
    return loss/len(loader.dataset), error/len(loader.dataset)

def load_best_model(model, train_cfg):
    model.load_state_dict(torch.load(
        train_cfg['dir_ckpts']+train_cfg['exp_name']+"/model.pt", map_location='cpu'))
    return model.cuda()
