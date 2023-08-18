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
from ..layers.upsampling_layer import DoubleConv, Up

from ..builder import HEADS

@HEADS.register_module()
class CoKeHeadMultiScale(BaseModule):

    def __init__(self, d_coke_feat=128, frozen=False):
        super(CoKeHeadMultiScale, self).__init__()
        self.upsample0 = DoubleConv(2048, 1024)
        self.upsample1 = Up(2048, 1024, 512)
        self.upsample2 = Up(1024, 512, 256)
        self.upsample3 = Up(512, 256, 256)

        self.out_layer_d2 = nn.Sequential(conv1x1(512, d_coke_feat), conv1x1(d_coke_feat, d_coke_feat))
        self.out_layer_d1 = nn.Sequential(conv1x1(256, d_coke_feat), conv1x1(d_coke_feat, d_coke_feat))
        self.out_layer = nn.Sequential(conv1x1(256, d_coke_feat), conv1x1(d_coke_feat, d_coke_feat))
        if frozen:
            self._freeze()

    def _freeze(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, features):
        output = {}

        feat_d3 = self.upsample0(features[3])
        feat_d2 = self.upsample1(feat_d3, features[2])
        feat_d1 = self.upsample2(feat_d2, features[1]), 
        features = self.upsample3(feat_d1, features[0])
        
        coke_features_d2 = F.interpolate(self.out_layer_d2(feat_d2), scale_factor=4, mode='bilinear', align_corners=True)
        coke_features_d1 = F.interpolate(self.out_layer_d1(feat_d1), scale_factor=2, mode='bilinear', align_corners=True)
        coke_features = self.out_layer(features)

        output.update(dict(features=features, 
                            coke_features=coke_features,
                            coke_features_d1=coke_features_d1,
                            coke_features_d2=coke_features_d2))

        return output


    