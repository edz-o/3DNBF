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
class CoKeHeadHRNet(BaseModule):

    def __init__(self, in_channels=480, hidden_channels=512, d_coke_feat=128, frozen=False):
        super(CoKeHeadHRNet, self).__init__()
        # self.transform = DoubleConv(in_channels, hidden_channels)
        # self.transform_pare = DoubleConv(in_channels, 256)

        self.out_layer_pare = conv1x1(in_channels, 256)
        self.out_layer = conv1x1(in_channels, d_coke_feat)
        if frozen:
            self._freeze()

    def _freeze(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, features):
        output = {}

        # x = self.transform(features)
        coke_features = self.out_layer(features)
        features = self.out_layer_pare(features)

        output.update(dict(features=features, 
                            coke_features=coke_features))

        return output


    