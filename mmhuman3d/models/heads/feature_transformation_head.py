# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from functools import partial
import torch
import numpy as np
import torch.nn as nn
from loguru import logger
import torch.nn.functional as F

from mmcv.runner.base_module import BaseModule

# from ...core.config import SMPL_MEAN_PARAMS
from ..layers.coattention import CoAttention
from ...utils.geometry import rot6d_to_rotmat, get_coord_maps
from ...utils.kp_utils import get_smpl_neighbor_triplets
from ..layers.softargmax import softargmax2d, get_heatmap_preds
from ..layers import LocallyConnected2d, KeypointAttention, interpolate
from ..layers.non_local import dot_product
from ..backbones.resnet import conv3x3, conv1x1, BasicBlock
from ..layers.upsampling_layer import DoubleConv, Up

from ..builder import HEADS

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super(Block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.norm = nn.LayerNorm(out_channels)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.act(x)
        return x

@HEADS.register_module()
class FeatureTransformationHead(BaseModule):

    def __init__(self, d_feat_in=16, d_coke_feat=16, frozen=False):
        super(FeatureTransformationHead, self).__init__()
        self.d_feat = d_feat_in
        # norm_layer = nn.LayerNorm
        # act_layer = nn.GELU # partial(nn.ReLU, inplace=True)
        self.layers = nn.Sequential(
            Block(self.d_feat, self.d_feat//2, 1, 1, 0, True),
            Block(self.d_feat//2, self.d_feat//2, 1, 1, 0, True),
            conv1x1(self.d_feat//2, d_coke_feat, bias=True),
        )
        self.apply(self._init_weights)
        if frozen:
            self._freeze()

    def _freeze(self, freeze=False):
        if freeze:
            self.eval()
            for param in self.parameters():
                param.requires_grad = False
        else:
            self.train()
            for param in self.parameters():
                param.requires_grad = True

    def forward(self, features):
        output = {}
        
        coke_features = self.layers(features)

        output.update(dict(features=features, 
                            coke_features=coke_features))

        return output

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)



    