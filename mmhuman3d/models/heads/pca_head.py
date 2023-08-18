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



@HEADS.register_module()
class PCAHead(BaseModule):

    def __init__(self, pca, feat_shape=80, layer=2):
        super(PCAHead, self).__init__()
        # norm_layer = nn.LayerNorm
        # act_layer = nn.GELU # partial(nn.ReLU, inplace=True)

        self.layer = layer
        self.feat_shape = feat_shape
        pca = np.load(pca, allow_pickle=True)
        # self.pca = nn.Linear(pca['in_dim'], pca['out_dim'], bias=False)
        self.pca = nn.Conv2d(pca['in_dim'], pca['out_dim'], kernel_size=1, stride=1, padding=0, bias=False)
        self.pca.weight.data = torch.from_numpy(pca['weight'])[:, :, None, None]
        # self.pca.bias.data = torch.from_numpy(pca['bias'])

    def forward(self, features):
        outputs = {}
        features = self.pca(features[self.layer])
        features = F.interpolate(features, size=(self.feat_shape, self.feat_shape), mode='bilinear', align_corners=True)
        outputs['coke_features'] = features
        return outputs




    