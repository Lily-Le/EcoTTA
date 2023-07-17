# -*- coding: utf-8 -*-
import torch
from torch.nn.modules import Module
from torch.nn import functional as F
import torch.nn as nn
from robustbench.model_zoo.enums import ThreatModel
from robustbench.utils import load_model

base_model = load_model('Hendrycks2020AugMix_WRN', './robust_models', 'cifar100', ThreatModel.corruptions).cuda()

class simplify_wideresnet40(Module):
    def __init__(self, model):
        super().__init__()
        # input_conv (in Figure3 in the paper)
        self.conv1 = model.conv1
        # encoder
        self.b1_l0 = model.block1.layer[0]
        self.b1_l1 = model.block1.layer[1]
        self.b1_l2 = model.block1.layer[2]
        self.b1_l3 = model.block1.layer[3]
        self.b1_l4 = model.block1.layer[4]
        self.b1_l5 = model.block1.layer[5]
        self.b2_l0 = model.block2.layer[0]
        self.b2_l1 = model.block2.layer[1]
        self.b2_l2 = model.block2.layer[2]
        self.b2_l3 = model.block2.layer[3]
        self.b2_l4 = model.block2.layer[4]
        self.b2_l5 = model.block2.layer[5]
        self.b3_l0 = model.block3.layer[0]
        self.b3_l1 = model.block3.layer[1]
        self.b3_l2 = model.block3.layer[2]
        self.b3_l3 = model.block3.layer[3]
        self.b3_l4 = model.block3.layer[4]
        self.b3_l5 = model.block3.layer[5]
        self.module_list = [self.b1_l0,self.b1_l1,self.b1_l2,self.b1_l3,self.b1_l4,self.b1_l5,\
                        self.b2_l0,self.b2_l1,self.b2_l2,self.b2_l3,self.b2_l4,self.b2_l5,\
                        self.b3_l0,self.b3_l1,self.b3_l2,self.b3_l3,self.b3_l4,self.b3_l5]
        self.input_depth_list = [16, 32, 32, 32, 32, 32, \
                                 32, 64, 64, 64, 64, 64, \
                                 64, 128, 128, 128, 128, 128, 128]
        self.input_hight_list = [32, 32, 32, 32, 32, 32,\
                                 32, 16, 16, 16, 16, 16,\
                                 16,  8,  8,  8,  8,  8,  8]
        # classifier
        self.classifier = classifier(model.bn1, model.relu,model.fc, model.nChannels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.b1_l0(out)
        out = self.b1_l1(out)
        out = self.b1_l2(out)
        out = self.b1_l3(out)
        out = self.b1_l4(out)
        out = self.b1_l5(out)
        out = self.b2_l0(out)
        out = self.b2_l1(out)
        out = self.b2_l2(out)
        out = self.b2_l3(out)
        out = self.b2_l4(out)
        out = self.b2_l5(out)
        out = self.b3_l0(out)
        out = self.b3_l1(out)
        out = self.b3_l2(out)
        out = self.b3_l3(out)
        out = self.b3_l4(out)
        out = self.b3_l5(out)
        out = self.classifier(out)
        return out

class classifier(Module):
    def __init__(self, bn1, relu, fc, nChannels):
        super().__init__()
        self.bn1 = bn1
        self.relu = relu
        self.fc = fc
        self.nChannels = nChannels

    def forward(self, x):
        out = self.relu(self.bn1(x))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        out = self.fc(out)
        return out

simplified_model = simplify_wideresnet40(base_model)

"""##  Attach meta networks"""



class conv_block(Module):
    def __init__(self, in_plane, out_plane, kernel_size=3, stride=1):
        super().__init__()
        padding = 1 if kernel_size == 3 else 0
        self.conv = nn.Conv2d(in_plane, out_plane, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_plane)
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))
        # return F.relu(self.conv(x))
#
class build_meta_block(Module):
    def __init__(self, in_out_depth_s, in_out_hight_s):
        super().__init__()
        self.in_out_depth_s = in_out_depth_s
        self.in_out_hight_s = in_out_hight_s
        self.meta_bn = nn.BatchNorm2d(in_out_depth_s[1]) #nn.Identity() #nn.Identity() #nn.BatchNorm2d(in_out_depth_s[1]) #nn.Identity()
        self.conv_block = conv_block(in_out_depth_s[0], in_out_depth_s[1], kernel_size=3, \
                                                stride=in_out_hight_s[0]//in_out_hight_s[1])
    def forward(self, x):
        out = self.conv_block(x)
        return out

class one_part_of_networks(Module):
    def __init__(self, original_part, meta_part):
        super().__init__()
        self.original_part = original_part
        self.meta_part = meta_part
        self.btsloss = None
        self.cal_mseloss = False

    def forward(self, x):
        # See Algorithm 1 in the paper (page13)
        if not self.cal_mseloss:
            out1 = self.original_part(x)
            out2 = self.meta_part.meta_bn(out1)
            out3 = self.meta_part(x)
            out = out2 + out3
        else:
            x = x.detach()
            out1 = self.original_part(x)
            out2 = self.meta_part.meta_bn(out1)
            out3 = self.meta_part(x)
            out = out2 + out3
            loss = nn.L1Loss(reduction='none')
            self.btsloss = loss(out, out1.detach()).mean()
        return out

def attach_meta_networks(simplified_model, K=5):
    # Set the number of blocks of each partition (Table 13 in the paper).
    if K==4:
        num_blocks = [3,3,6,6]
    elif K==5:
        num_blocks = [3,3,3,3,6]
    else: ValueError

    # Get necessary informations to build convolution layers of meta networks,
    # such as, the number of channels of input and output feature from the original networks.
    in_out_depth_s = [] # channels / dimensions
    in_out_hight_s = [] # width / hight
    start_module = 0
    for l in num_blocks:
        in_out_depth_s.append((simplified_model.input_depth_list[start_module], simplified_model.input_depth_list[start_module+l]))
        in_out_hight_s.append((simplified_model.input_hight_list[start_module], simplified_model.input_hight_list[start_module+l]))
        start_module = start_module+l

    class ecotta_networks(Module):
        def __init__(self, simplified_model, num_blocks, in_out_depth_s, in_out_hight_s):
            super().__init__()
            self.conv1 = simplified_model.conv1
            encoders = []
            start_module = 0
            for l in num_blocks:
                encoder = nn.Sequential(*simplified_model.module_list[start_module: start_module+l])
                encoders.append(encoder)
                start_module = start_module+l
            self.encoders = nn.Sequential(*encoders)
            self.classifier = simplified_model.classifier

            self.meta_parts = []
            for i in range(len(num_blocks)):
                meta_part = build_meta_block(in_out_depth_s[i], in_out_hight_s[i])
                self.encoders[i] = one_part_of_networks(self.encoders[i], meta_part)
                self.meta_parts.append(meta_part)

        def forward(self, x):
            out = self.conv1(x)
            out = self.encoders(out)
            out = self.classifier(out)
            return out

    # Return whole networks including original and meta networks
    return ecotta_networks(simplified_model, num_blocks, in_out_depth_s, in_out_hight_s)

ecotta_networks = attach_meta_networks(simplified_model, K=5)
for param in ecotta_networks.parameters():
    param.requires_grad = False
for meta_part in ecotta_networks.meta_parts:
    for param in meta_part.parameters():
        param.requires_grad = True

