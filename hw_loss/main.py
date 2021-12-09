import torch
import torch.utils.data as D
import sys
import argparse
import statistics
import os
from tensorboardX import SummaryWriter
sys.path.insert(0, os.getcwd())

from hw_loss.dataset import *
from hw_loss.utils import *
from hw_loss.trainer import *
from hw_loss.networks import *

def main():

    parser = argparse.ArgumentParser(description='Pytorch Experiment')
    parser.add_argument('--config', metavar='C', type=str, default='our_nn/config.yaml',
                        help='Specify config file', dest='cfg')
    parser.add_argument('--gpu', type=int, default=0, metavar='N',
                        help='input the gpu id (default: 1)')
    parser.add_argument('--para', type=str, default='pip', help='Specify paradigm')
    args = parser.parse_args()

    torch.cuda.set_device(0)

    dataset_cfg, model_cfg, train_cfg = ConfigLoader.load_all_cfg(args.cfg)

    train_ds, val_ds, test_ds = get_ds(args.para)

    test_loss_list = []
    test_err_list = []
    for run in range(train_cfg['num_runs']):
        change_seed(run)
        train_loader = D.DataLoader(
            train_ds, batch_size=train_cfg['batch_size'], shuffle=True, drop_last=True)
        val_loader = D.DataLoader(
            val_ds, batch_size=train_cfg['batch_size'], shuffle=True, drop_last=False)
        test_loader = D.DataLoader(
            test_ds, batch_size=train_cfg['batch_size'], shuffle=True, drop_last=False)

        model = get_model(model_cfg)
        model = model.cuda()
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Num of params:{}'.format(total_params))

        writer = SummaryWriter(train_cfg['dir_log']+train_cfg['exp_name'])
        train(model, writer, run, train_cfg, train_loader, val_loader)
        model = load_best_model(model, train_cfg)
        test_loss, test_err = eval(model, train_cfg, test_loader)
        print('Test loss :{}'.format(test_loss))
        print('Test err :{}'.format(test_err))
        writer.add_scalar(str(run)+'/test_loss', test_loss, 0)
        writer.add_scalar(str(run)+'/test_err', test_err, 0)
        test_loss_list.append(test_loss)
        test_err_list.append(test_err)

    writer.add_scalar('test_loss_mean', statistics.mean(test_loss_list), 0)
    writer.add_scalar('test_loss_std', statistics.stdev(test_loss_list), 0)
    writer.add_scalar('test_err_mean', statistics.mean(test_err_list), 0)
    writer.add_scalar('test_err_std', statistics.stdev(test_err_list), 0)
    print('{} runs loss: mean: {} std: {}'.format(
        train_cfg['num_runs'], statistics.mean(test_loss_list), statistics.stdev(test_loss_list)))
    print('{} runs err: mean: {} std: {}'.format(
        train_cfg['num_runs'], statistics.mean(test_err_list), statistics.stdev(test_err_list)))

if __name__ == '__main__':
    main()
