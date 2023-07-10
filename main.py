# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

import os
import sys
import argparse
import time
import math
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from robustbench.data import load_cifar100c
from robustbench.utils import clean_accuracy as accuracy
from conf import settings
from utils import get_training_dataloader_ecotta, get_test_dataloader, entropy_minmization
from ecotta_sample_code import ecotta_networks,set_cal_mseloss,base_model
from conf_tta import cfg, load_cfg_fom_args
from my_utils import Logger
import pandas as pd      


def adapt():
    start = time.time()
    tta_path = os.path.dirname(checkpoint_path)+'/tta'
    if not os.path.exists(tta_path):
        os.makedirs(tta_path)
    tta_log_path = tta_path + f'/{args.tta_optim}_bn{args.tta_bn}_bs{args.b}_ttalr{args.lr_tta}_lamb{args.lambda_reg}.log'
    sys.stdout = Logger(tta_log_path, sys.stdout)
    print('--------------   begin tta   ------------------------')

    for name,paras in net.named_modules():
        if 'meta_part' in name:
            paras.train()

    df0 = pd.DataFrame(data=[vars(args).values()], columns=vars(args).keys())
    df_all=pd.DataFrame()
    for severity in cfg.CORRUPTION.SEVERITY:
        err_mean = 0.
        for i_c, corruption_type in enumerate(cfg.CORRUPTION.TYPE):
            df_ = df0.copy(deep=True)

            x_test, y_test = load_cifar100c(cfg.CORRUPTION.NUM_EX,
                                           severity, cfg.DATA_DIR, False,
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
                set_cal_mseloss(ecotta_networks, True)
                outputs = ecotta_networks(x_curr)
                loss_ent = entropy_minmization(outputs,e_margin=args.e_margin)

                loss_reg_all = 0.
                gamma = 2 ## What's gamma used for?  In the paper, lambda is 0.5
                for i, encoder in enumerate(ecotta_networks.encoders):
                    reg_loss = encoder.btsloss * gamma
                    loss_reg_all += reg_loss
                    # try:
                    #     reg_loss.backward()
                    # except:
                    #     print(i)
                loss = loss_ent+args.lambda_reg*loss_reg_all
                loss.backward()
                set_cal_mseloss(ecotta_networks, False)

                optimizer_TTA.step()
                optimizer_TTA.zero_grad()
                acc += (outputs.max(1)[1] == y_curr).float().sum()/ x_test.shape[0]
            err = 1 - acc
            err_mean += err
            print('not resetting model')
            # logger.info(f"error % [{corruption_type}{severity}]: {err:.2%}")
            print(f"err % [{corruption_type}{severity}]: {err:.2%}")
            df_['corruption type'] = corruption_type
            df_['severity'] = severity
            df_['adaptation error'] = err.item()
            # df_all = df_all.append(df_, ignore_index=True)
            df_all = pd.concat([df_all, df_], ignore_index=True)

        err_mean /= len(cfg.CORRUPTION.TYPE)
        df_=df0.copy(deep=True)
        df_['corruption type']='mean'
        df_['severity']=severity
        df_['adaptation error']=err_mean.item()
        df_all = pd.concat([df_all, df_], ignore_index=True)
        # logger.info(f"error % [{severity}]: {err_mean:.2%}")
        print(f"err mean % [{severity}]: {err_mean:.2%}")

    print(f'Time used: {time.time()-start:.2f}s')
    tta_result_path = 'tta_results/'+tta_log_path.replace('/','-').replace('log','csv')
    df_all.to_csv(tta_result_path, index=False)


def warmup_train(epoch):

    start = time.time()

    if args.pretrain_bn: #pretrain_bn: running bn for the source model during warmup process
        net.train()
    else:
        net.eval() #  frozen bn statistics for the source model during warmup process

    # Running bn statistics for meta network
    for name,paras in net.named_modules():
        if 'meta_part' in name:
            paras.train()

    correct=0.0
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):

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

        n_iter = (epoch - 1) * len(cifar100_training_loader) + batch_index + 1

        
        for name,param in net.named_parameters():
            if 'meta_part' in name:
                writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
                try:
                    # print(param.grad)
                    writer.add_histogram(name + '/grad', param.grad.clone().cpu().data.numpy(), epoch)
                except:
                    print(f'{name} no grad')
                    # pass
                    # if epoch !=1:
                    #     print(f'{name} no grad')
        #
        #             if 'weight' in name:
        #                 writer.add_scalar('Meta_parts/grad_norm2_weights', para.grad.norm(), n_iter)
        #             if 'bias' in name:
        #                 writer.add_scalar('Meta_parts/grad_norm2_bias', para.grad.norm(), n_iter)
        #             # record the histogram of running mean and var in BN layer
        #             if 'bn' in name:
        #                 writer.add_histogram('bn_mean', para.running_mean,)
        #                 writer.add_histogram('bn_std', para.running_var, )
        # for name, param in net.named_parameters():
        #     writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
        #     writer.add_histogram(name + '/grad', param.grad.cLone().cpu().data.numpy(), epoch)
        optimizer.step()

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tAcc: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            correct.item() / float(args.b * (batch_index + 1)),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(cifar100_training_loader.dataset)
        ))


        #update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)
        writer.add_scalar('Train/acc', correct.item() / float(args.b * (batch_index + 1)), n_iter)
        # if epoch <= args.warm:
        #     warmup_scheduler.step()

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
    for (images, labels) in cifar100_test_loader:

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
        test_loss / len(cifar100_test_loader.dataset),
        correct.float() / len(cifar100_test_loader.dataset),
        finish - start
    ))
    print()

    #add informations to tensorboard
    if tb:
        writer.add_scalar('Test/Average loss', test_loss / len(cifar100_test_loader.dataset), epoch)
        writer.add_scalar('Test/Accuracy', correct.float() / len(cifar100_test_loader.dataset), epoch)

    return correct.float() / len(cifar100_test_loader.dataset)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, default='wideresnet', help='net type')
    parser.add_argument('-gpu', action='store_true', default=True, help='use gpu or not')
    parser.add_argument('-b', type=int, default=64, help='batch size for dataloader')
    # parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=5e-2, help='initial learning rate')
    parser.add_argument('-lr_tta', type=float, default=2.5e-5, help='initial learning rate')

    parser.add_argument('-warmup_checkpoint', type=str, default='checkpoint_pretrain/wideresnet/original_meta_bnTrue_bs64_lr1.5e-2/wideresnet-10-regular.pth', help='checkpoint to load, needed for tta mode')
    parser.add_argument('-e_margin', type=float, default=math.log(100)*0.40, help='entropy margin E_0 in Eqn. (3) for filtering reliable samples') # default value for cifar100
    parser.add_argument('-lambda_reg', type=float, default=1, help='importance of Regulation item')
    parser.add_argument('-mode', type=str, default='pretrain',help='pretrain:warmup phase; tta:tta phase ; both:warmup+tta')#)
    parser.add_argument('-pretrain_bn',action='store_true', default=True, help='whether to freeze the source bn statistics in pretrain phase')
    parser.add_argument('-tta_optim',type=str, default='new_optim') # original_optim: tta optimizer inherit the statedict of warmup optimizer
    parser.add_argument('-tta_bn', action='store_true', default=False, help='whether to freeze the source bn statistics in tta phase')
    #other params: conf/global_settings.py, conf_tta.py

    args = parser.parse_args()

    # net = get_network(args)
    net = ecotta_networks
    result_dir=os.path.join(args.net, f'original_meta_bn{args.pretrain_bn}_bs{args.b}_lr{args.lr}')
    log_dir = 'checkpoint_pretrain'+f'/{result_dir}'
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    # data preprocessing:
    # transform for training data: colorjitter, gaussianblur, grayscale
    cifar100_training_loader = get_training_dataloader_ecotta(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )
    # transform for test data: to tensor
    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )

    loss_function = nn.CrossEntropyLoss()
    # parameters that require grad: defined in sample code
    optimizer = optim.SGD([params for params in net.parameters() if params.requires_grad], lr=args.lr, momentum=0.9) #, weight_decay=5e-4)
    optimizer_TTA = torch.optim.SGD([params for params in net.parameters() if params.requires_grad], args.lr_tta, momentum=0.9)
    # train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(cifar100_training_loader)
    # warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, result_dir)
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)

    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, result_dir))

    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0

    if (args.mode == 'pretrain') or (args.mode =='both'): #prertrain: warmup training; both: first warmup then tta
        sys.stdout = Logger(log_dir + '/logs_pretrain.log', sys.stdout)

        print('----------------start pretrain-----------------')
        for epoch in range(1, settings.EPOCH + 1):
            # if epoch > args.warm:
            #     train_scheduler.step(epoch)
            warmup_train(epoch)
            acc = eval_training(epoch)

            #start to save best performance model after learning rate decay to 0.01
            if epoch > settings.MILESTONES[1] and best_acc < acc:
                weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='best')
                print('saving weights file to {}'.format(weights_path))
                torch.save({'net':net.state_dict(),'optim':optimizer.state_dict()}, weights_path)
                best_acc = acc
                continue

            if not epoch % settings.SAVE_EPOCH:
                weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular')
                print('saving weights file to {}'.format(weights_path))
                torch.save({'net':net.state_dict(),'optim':optimizer.state_dict()}, weights_path)

    if args.mode == 'both':
        if args.tta_optim == 'original_optim':
            optimizer_TTA.load_state_dict(optimizer.state_dict())
        if args.tta_bn:
            net.train()
        else:
            net.eval()
        adapt()
    elif args.mode == 'tta':
        # checkpoint_path = 'checkpoint_pretrain/wideresnet/original_meta_bnTrue_bs64_lr1.5e-2/wideresnet-10-regular.pth'
        checkpoint_path = args.warmup_checkpoint
        ckpt = torch.load(checkpoint_path)
        net.load_state_dict(ckpt['net'])
        if 'bnFalse' in checkpoint_path.split('/')[-2]:
            args.pretrain_bn=False
        else:
            args.pretrain_bn=True
        if args.tta_optim == 'original_optim':
            optimizer_TTA.load_state_dict(ckpt['optim'])
        if args.tta_bn:
            net.train()
        else:
            net.eval()

        adapt()
    # else:
    #     raise Exception('mode not supported')
    writer.close()