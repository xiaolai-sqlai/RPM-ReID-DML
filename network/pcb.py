from __future__ import absolute_import

import math
import random
import copy
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from einops import rearrange
from functools import reduce

from network.layer import BatchDrop, BatchErasing
from network.regnet_y import RegNetY, ConvX


class GlobalAvgPool2d(nn.Module):
    def __init__(self, p=1):
        super(GlobalAvgPool2d, self).__init__()
        self.p = p
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        out = x.pow(self.p)
        out = self.gap(out)
        return out.pow(1/self.p)


class GlobalMaxPool2d(nn.Module):
    def __init__(self, p=1):
        super(GlobalMaxPool2d, self).__init__()
        self.p = p
        self.gap = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        out = x.pow(self.p)
        out = self.gap(out)
        return out.pow(1/self.p)


class PCB(nn.Module):
    def __init__(self, decoder, num_classes=751, num_part=2, num_stripe=1, feat_num=0, std=0.1, net="regnet_y_1_6gf", erasing=0.0, h=384, w=128, use_global=True):
        super(PCB, self).__init__()
        self.num_part = num_part
        self.num_stripe = num_stripe
        self.feat_num = feat_num
        self.h = h
        self.w = w
        self.use_global = use_global
        if self.training:
            self.erasing = nn.Identity()
            if erasing > 0:
                self.erasing = BatchErasing(smax=erasing)

        if net == "regnet_y_1_6gf":
            base = RegNetY(dims=[64,128,256,512], layers=[6,12,27,6], ratio=1.0, drop_path_rate=0.10)
            path = "pretrain/checkpoint_regnet_y_1_6gf.pth"
        elif net == "regnet_y_1_6gf_prelu":
            base = RegNetY(dims=[64,128,256,512], layers=[6,12,27,6], ratio=1.0, act_type="prelu", drop_path_rate=0.10)
            path = "pretrain/checkpoint_regnet_y_1_6gf_prelu.pth"

        old_checkpoint = torch.load(path)["state_dict"]
        new_checkpoint = dict()
        for key in old_checkpoint.keys():
            if key.startswith("module."):
                new_checkpoint[key[7:]] = old_checkpoint[key]
            else:
                new_checkpoint[key] = old_checkpoint[key]
        base.load_state_dict(new_checkpoint)

        self.stem = base.first_conv
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3

        base.layer4[0].main[1].conv.stride = (1, 1)
        base.layer4[0].skip[0].conv.stride = (1, 1)

        self.branch_1 = copy.deepcopy(nn.Sequential(base.layer4, base.head))
        self.branch_2 = copy.deepcopy(nn.Sequential(base.layer4, base.head))

        self.pool_list = nn.ModuleList()
        self.feat_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        self.class_list = nn.ModuleList()

        if self.use_global:
            self.pool_list.append(GlobalAvgPool2d(p=1))
            self.pool_list.append(GlobalMaxPool2d(p=1))
            for i in range(2):
                if self.feat_num == 0:
                    feat_num = 1024
                    feat = nn.Identity()
                else:
                    feat = nn.Linear(1024, feat_num, bias=False)
                    init.kaiming_normal_(feat.weight)
                self.feat_list.append(feat)
                bn = nn.BatchNorm1d(feat_num)
                if i == 1:
                    bn.bias.requires_grad = False
                init.normal_(bn.weight, mean=1.0, std=std)
                init.normal_(bn.bias, mean=0.0, std=std)
                self.bn_list.append(bn)

                linear = nn.Linear(feat_num, num_classes, bias=False)
                init.normal_(linear.weight, std=0.001)
                self.class_list.append(linear)

        for i in range(self.num_part):
            if "max" in decoder:
                self.pool_list.append(GlobalMaxPool2d(p=1))
            elif "avg" in decoder:
                self.pool_list.append(GlobalAvgPool2d(p=1))
            if self.feat_num == 0:
                feat_num = 1024
                feat = nn.Identity()
            else:
                feat = nn.Linear(1024, feat_num, bias=False)
                init.kaiming_normal_(feat.weight)
            self.feat_list.append(feat)
            bn = nn.BatchNorm1d(feat_num)
            init.normal_(bn.weight, mean=1.0, std=std)
            init.normal_(bn.bias, mean=0.0, std=std)
            bn.bias.requires_grad = False
            self.bn_list.append(bn)

            linear = nn.Linear(feat_num, num_classes, bias=False)
            init.normal_(linear.weight, std=0.001)
            self.class_list.append(linear)

    def forward(self, x):
        if self.training:
            x = self.erasing(x)

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)

        x_chunk = []
        if self.use_global:
            x_chunk = [x1, x1]
        for i, part in enumerate(torch.chunk(x2, dim=2, chunks=self.num_part)):
            x_chunk.append(part)

        pool_list = []
        feat_list = []
        bn_list = []
        class_list = []

        for i in range(self.num_part + 2 if self.use_global else self.num_part):
            pool = self.pool_list[i](x_chunk[i]).flatten(1)
            pool_list.append(pool)
            feat = self.feat_list[i](pool)
            feat_list.append(feat)
            bn = self.bn_list[i](feat)
            bn_list.append(bn)
            feat_class = self.class_list[i](bn)
            class_list.append(feat_class)

        if self.training:
            return class_list, bn_list[:2] if self.use_global else []
        return bn_list, 

