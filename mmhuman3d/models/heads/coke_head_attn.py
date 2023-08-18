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
from ..layers.upsampling_layer import DoubleConv, Up, AttnUp
from ..backbones.vision_transformer import Attention, PreNorm, FeedForward

from ..builder import HEADS

@HEADS.register_module()
class CoKeHeadAttn(BaseModule):

    def __init__(self, d_coke_feat=128, hidden_dim = 256, num_heads=8, n_attn_layer=3, frozen=False):
        super(CoKeHeadAttn, self).__init__()

        
        self.upsample0 = DoubleConv(2048, 1024)
        self.upsample1 = Up(2048, 1024, 512)
        self.upsample2 = AttnUp(1024, 512, 256)
        self.upsample3 = Up(512, hidden_dim, hidden_dim)

        # self.attn_layers = nn.ModuleList([])
        # for _ in range(n_attn_layer):
        #     self.attn_layers.append(nn.ModuleList([
        #     PreNorm(hidden_dim, Attention(hidden_dim, num_heads=num_heads)),
        #     PreNorm(hidden_dim, FeedForward(hidden_dim, hidden_dim, dropout=0.))]
        #     ))
                        
        self.out_layer = conv1x1(hidden_dim, d_coke_feat)
        if frozen:
            self._freeze()

    def _freeze(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, features):
        output = {}

        features=self.upsample3(
                self.upsample2(
                self.upsample1(
                self.upsample0(
                    features[3]), 
                    features[2]), 
                    features[1]), 
                    features[0])
        
        # b, c, h, w = features.shape
        # features = features.permute(0, 2, 3, 1).view(b, h*w, c)
        # for attn, ff in self.attn_layers:
        #     features = attn(features) + features
        #     features = ff(features) + features
        # features = features.view(b, h, w, c).permute(0, 3, 1, 2)
        coke_features = self.out_layer(features)

        output.update(dict(features=features, 
                            coke_features=coke_features))

        return output


    