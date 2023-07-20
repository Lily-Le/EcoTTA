# train.py
#!/usr/bin/env	python3

""" train network using pytorch
Reference Code: [pytorch-cifar100](https://github.com/weiaicunzai/pytorch-cifar100)   
[EATA](https://github.com/mr-eggplant/EATA/blob/main)  
[TENT](https://github.com/DequanWang/tent)  

"""

import os
import sys
import argparse
import time
import math

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from robustbench.data import load_cifar100c, load_cifar10c
# from conf import settings
from utils import get_training_dataloader, get_test_dataloader, entropy_minmization,set_cal_mseloss,WarmUpLR,Logger
from ecotta_models.wideresnet40 import ecotta_networks as ecotta_networks_wrs40
from ecotta_models.wideresnet28 import ecotta_networks as ecotta_networks_wrs28
from conf import cfg

# import wandb

def adapt():
    start = time.time()

    for name,paras in net.named_modules():
        if 'meta_part' in name:
            paras.train()


    if args.dataset =='cifar100':
        load_corruption_dataset = load_cifar100c
        e_margin = math.log(100)*args.e_margin
    elif args.dataset =='cifar10':
        load_corruption_dataset = load_cifar10c
        e_margin = math.log(10) * args.e_margin
    else:
        raise NotImplementedError
    for severity in cfg.CORRUPTION.SEVERITY:
        err_mean = 0.
        for i_c, corruption_type in enumerate(cfg.CORRUPTION.TYPE):
            x_test, y_test = load_corruption_dataset(cfg.CORRUPTION.NUM_EX,
                                           severity, args.data_path, False,
                                           [corruption_type])
            x_test, y_test = x_test.cuda(), y_test.cuda()
            acc = 0.
            n_batches = math.ceil(x_test.shape[0] / args.b)

            device = x_test.device
            for counter in range(n_batches):
                optimizer_TTA.zero_grad()
                x_curr = x_test[counter * args.b:(counter + 1) *
                                                args.b].to(device)
                y_curr = y_test[counter * args.b:(counter + 1) *
                                                args.b].to(device)
                set_cal_mseloss(net, True)

                n_iter = i_c*len(cfg.CORRUPTION.TYPE)+counter
                
                outputs = net(x_curr)
                loss_reg_all = 0.
                gamma = args.lambda_reg 
                for i, encoder in enumerate(net.encoders):
                    reg_loss = encoder.btsloss * gamma
                    reg_loss.backward()
                    writer.add_scalar(f'loss_reg_{i}', reg_loss.item(), n_iter)
                    loss_reg_all += reg_loss.item()
                writer.add_scalar('loss_reg_all', loss_reg_all, n_iter)

                optimizer_TTA.step()
                optimizer_TTA.zero_grad()

                set_cal_mseloss(net, False)
                outputs = net(x_curr)
                loss_ent = entropy_minmization(outputs,e_margin=e_margin)
                loss_ent.backward()
                writer.add_scalar('loss_ent', loss_ent.item(), n_iter)
                optimizer_TTA.step()
                optimizer_TTA.zero_grad()
                acc += (outputs.max(1)[1] == y_curr).float().sum()/ x_test.shape[0]
                acc_curr = (outputs.max(1)[1] == y_curr).float().sum()/ x_curr.shape[0]
                writer.add_scalar('acc_curr', acc_curr.item(), n_iter)

            err = 1 - acc
            err_mean += err
            print('not resetting model')
            print(f"err % [{corruption_type}{severity}]: {err:.2%}")



        err_mean /= len(cfg.CORRUPTION.TYPE)

        # logger.info(f"error % [{severity}]: {err_mean:.2%}")
        print(f"err mean % [{severity}]: {err_mean:.2%}")

    print(f'Time used: {time.time()-start:.2f}s')


def warmup_train(epoch):

    start = time.time()
    # running bn for the source model during warmup process
    net.train()

    # Running bn statistics for meta network
    for name,paras in net.named_modules():
        if 'meta_part' in name:
            paras.train()

    correct=0.0
    for batch_index, (images, labels) in enumerate(training_loader):

        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        # pretrain: only calculate cross entropy loss
        set_cal_mseloss(net, False)
        outputs = net(images)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

        loss = loss_function(outputs, labels)
        # loss2 = loss_function(outputs2, labels)
        loss.backward()

        n_iter = (epoch - 1) * len(training_loader) + batch_index + 1

        
        for name,param in net.named_parameters():
            if 'meta_part' in name:
                writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
                try:
                    # print(param.grad)
                    writer.add_histogram(name + '/grad', param.grad.clone().cpu().data.numpy(), epoch)
                except:
                    # print(f'{name} no grad')
                    pass
                    
        optimizer.step()

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tAcc: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            correct.item() / float(args.b * (batch_index + 1)),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(training_loader.dataset)
        ))


        #update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)
        writer.add_scalar('Train/acc', correct.item() / float(args.b * (batch_index + 1)), n_iter)
        if epoch <= args.warm:
           warmup_scheduler.step()

    # for name, param in net.named_parameters():
    #     if 'meta_parts' in name:
    #         layer, attr = os.path.splitext(name)
    #         attr = attr[1:]
    #         writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

@torch.no_grad()
def eval_training(epoch=0, tb=True):

    start = time.time()

    net.eval()
    test_loss = 0.0 # cost function error
    correct = 0.0
    for (images, labels) in test_loader:

        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()
        # correct2 += preds2.eq(labels).sum()

    finish = time.time()
    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(test_loader.dataset),
        correct.float() / len(test_loader.dataset),
        finish - start
    ))
    print()

    #add informations to tensorboard
    if tb:
        writer.add_scalar('Test/Average loss', test_loss / len(test_loader.dataset), epoch)
        writer.add_scalar('Test/Accuracy', correct.float() / len(test_loader.dataset), epoch)

    return correct.float() / len(test_loader.dataset)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, default='wideresnet28', help='net type')
    parser.add_argument('--dataset', type=str, default='cifar10', help='net type')
    parser.add_argument('--data_path', type=str, default='./data', help='path to dataset')
    parser.add_argument('--gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('--b', type=int, default=64, help='batch size for dataloader')
    parser.add_argument('--mode', type=str, default='pretrain',help='pretrain:warmup phase; tta:tta phase;')#)
    #--------If mode = pretrain
    parser.add_argument('--checkpoint_path', type=str, default='./checkpoint_pretrain/trainscheduler', help='path to save warmup model')
    parser.add_argument('--warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('--lr', type=float, default=5e-2, help='initial learning rate for warmup phase')
    parser.add_argument('--warmup_epoch', type=int, default=10, help='total epochs for warmup stage')
    parser.add_argument('--log_dir_pretrain', type=str, default='logs_pretrain/trainscheduler', help='log directory for Tensorboard log output')
    #--------If mode = tta
    parser.add_argument('--lr_tta', type=float, default=5e-3, help='initial learning rate')
    parser.add_argument('--warmup_checkpoint', type=str, default='./checkpoint_pretrain/wideresnet28/warmup_bs64_lr0.05/wideresnet28-10-regular.pth', help='checkpoint to load, needed for tta mode')
    parser.add_argument('--e_margin', type=float, default=0.40, help='entropy margin E_0 in Eqn. (3) for filtering reliable samples') # default value for cifar100
    parser.add_argument('--lambda_reg', type=float, default=0.25, help='importance of Regulation item')
    parser.add_argument('--log_dir_tta', type=str, default='logs_tta', help='log directory for Tensorboard log output')
    

    args = parser.parse_args()

    if args.net == 'wideresnet40':
        net = ecotta_networks_wrs40
    elif args.net.startswith('wideresnet28'):
        net = ecotta_networks_wrs28
    else:
        raise NotImplementedError

    net.cuda()
    for param in net.parameters():
        param.requires_grad = False
    for meta_part in net.meta_parts:
        for nm,param in meta_part.named_parameters():
            param.requires_grad = True



    result_dir = os.path.join(args.net,
        f'warmup_bs{args.b}_lr{args.lr}')

    # data preprocessing:
    # transform for training data: colorjitter, gaussianblur, grayscale
    training_loader= get_training_dataloader(
        args.dataset,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )
    # transform for test data: to tensor
    test_loader= get_test_dataloader(
        args.dataset,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )

    loss_function = nn.CrossEntropyLoss()
    # parameters that require grad: defined in sample code
    optimizer = optim.SGD([params for params in net.parameters() if params.requires_grad], lr=args.lr, momentum=0.9) #, weight_decay=5e-4)
    optimizer_TTA = torch.optim.SGD([params for params in net.parameters() if params.requires_grad], args.lr_tta, momentum=0.9)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3,6,8], gamma=0.2) #learning rate decay
    iter_per_epoch = len(training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    best_acc = 0.0

    if (args.mode == 'pretrain'): #prertrain: warmup training; both: first warmup then tta
        checkpoint_path = os.path.join(args.checkpoint_path, result_dir)
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        sys.stdout = Logger(checkpoint_path + '/logs_pretrain.log', sys.stdout)

        #use tensorboard
        if not os.path.exists(args.log_dir_pretrain):
            os.mkdir(args.log_dir_pretrain)
        checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')
        log_path = os.path.join(
            args.log_dir_pretrain, result_dir)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        writer = SummaryWriter(log_dir=log_path)

        print('----------------start pretrain-----------------')
 # running bn for the source model during warmup process
        net.train()

        for epoch in range(1, args.warmup_epoch + 1):
            if epoch > args.warm:
                train_scheduler.step(epoch)
            warmup_train(epoch)
            acc = eval_training(epoch)


            if not epoch % args.warmup_epoch:
                weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular')
                print('saving weights file to {}'.format(weights_path))
                torch.save({'net':net.state_dict(),'optim':optimizer.state_dict()}, weights_path)

    elif args.mode == 'tta':
        checkpoint_path = args.warmup_checkpoint
        ckpt = torch.load(checkpoint_path)


        net.load_state_dict(ckpt['net'])
        args.arch = checkpoint_path.split('/')[-2].split('_')[0]+checkpoint_path.split('/')[-2].split('_')[1]
            # for n in net.modules():
            #     if isinstance(n, nn.BatchNorm2d):
            #         n.running_var=None
            #         n.running_mean=None
        net.train()



        tta_path = os.path.dirname(checkpoint_path) + '/tta'
        if not os.path.exists(tta_path):
            os.makedirs(tta_path)

        tta_log_path = tta_path + f'/bs{args.b}_ttalr{args.lr_tta}_lamb{args.lambda_reg}_{args.e_margin}.log'
        sys.stdout = Logger(tta_log_path, sys.stdout)

        tta_tblog_path =  os.path.dirname(checkpoint_path).replace(args.checkpoint_path,args.log_dir_tta) + f'/bs{args.b}_ttalr{args.lr_tta}_lamb{args.lambda_reg}_{args.e_margin}'
        writer = SummaryWriter(log_dir=tta_tblog_path)

        print('--------------   begin tta   ------------------------')
        print(tta_tblog_path)
     
        adapt()
    else:
        raise NotImplementedError
   
    writer.close()