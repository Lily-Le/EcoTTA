""" helper function
Reference Code: [pytorch-cifar100](https://github.com/weiaicunzai/pytorch-cifar100)   
[TENT](https://github.com/DequanWang/tent)  
"""
import os
import sys


import torch
from torch.optim.lr_scheduler import _LRScheduler
import torchvision
import torchvision.transforms as transforms
from torchvision import transforms as T
from torch.utils.data import DataLoader

def entropy_minmization(outputs,e_margin):
    """Calculate entropy of the output of a batch of images.
    """
    # convert to probabilities
    entropys = softmax_entropy(outputs)
    # filter unreliable samples
    filter_ids_1 = torch.where(entropys < e_margin)
    # ids1 = filter_ids_1
    # ids2 = torch.where(ids1[0] > -0.1)
    entropys = entropys[filter_ids_1]
    loss = entropys.mean(0)
    return loss

def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    temprature = 1
    x = x/ temprature
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x

def set_cal_mseloss(networks, cal_mseloss:bool):
    for encoder in networks.encoders:
        encoder.cal_mseloss = cal_mseloss


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """
    def __init__(self, optimizer, total_iters, last_epoch=-1):

        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]

def get_training_dataloader(dataset,batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = T.Compose([
        T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.4),
        T.RandomApply([T.GaussianBlur(kernel_size=(3,3), sigma=(0.1, 2.0))], p=0.2),
        T.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
    # transforms.Normalize([0.5] * 3, [0.5] * 3)        # transforms.Normalize(mean, std)
    ])
    #cifar100_training = CIFAR100Train(path, transform=transform_train)
    if dataset=='cifar100':
        cifar_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
        cifar_training_loader = DataLoader(
            cifar_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    elif dataset=='cifar10':
        cifar_training = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        cifar_training_loader = DataLoader(
            cifar_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    else:
        raise ValueError('dataset name not found')

    return cifar_training_loader

def get_test_dataloader(dataset,batch_size=16, num_workers=2, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    # transforms.Normalize([0.5] * 3, [0.5] * 3)
        # transforms.Normalize(mean, std)
    ])
    #cifar100_test = CIFAR100Test(path, transform=transform_test)
    if dataset=='cifar100':
        cifar_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
        cifar_test_loader = DataLoader(
            cifar_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    elif dataset=='cifar10':
        cifar_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        cifar_test_loader = DataLoader(
            cifar_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size)
    else:
        raise ValueError('dataset name not found')

    return cifar_test_loader

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass
 
