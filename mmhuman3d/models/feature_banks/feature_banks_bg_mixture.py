import torch
from torch.autograd import Function
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn
import math

from mmcv.runner.base_module import BaseModule
from ..builder import FEATURE_BANKS
from mmhuman3d.utils.dist_utils import allgather_tensor
from mmhuman3d.utils.image_utils import one_hot
from .feature_banks import VectorAverageMeter
from loguru import logger

def to_mask(y, max_size):
    y_onehot = torch.zeros((len(y), max_size), dtype=torch.float32, device=y[0].device)
    for i in range(len(y)):
        y_onehot[i].scatter_(0, y[i].type(torch.long), 1)
    return y_onehot


def squared_euclidean_distance_matrix(
    pts1: torch.Tensor, pts2: torch.Tensor
) -> torch.Tensor:
    """
    Get squared Euclidean Distance Matrix
    Computes pairwise squared Euclidean distances between points
    Args:
        pts1: Tensor [M x D], M is the number of points, D is feature dimensionality
        pts2: Tensor [N x D], N is the number of points, D is feature dimensionality
    Return:
        Tensor [M, N]: matrix of squared Euclidean distances; at index (m, n)
            it contains || pts1[m] - pts2[n] ||^2
    """
    edm = torch.mm(-2 * pts1, pts2.t())
    edm += (pts1 * pts1).sum(1, keepdim=True) + (pts2 * pts2).sum(1, keepdim=True).t()
    return edm.contiguous()


@FEATURE_BANKS.register_module()
class Nearest3DMemoryBGMixtueManager(BaseModule):
    def __init__(
        self,
        inputSize,
        num_pos,
        bg_bank_size,
        num_orient=3,
        T=0.07,
        momentum=0.5,
        Z=None,
        max_groups=-1,
        num_noise=-1,
        feat_normalization=False,
    ):
        """
        Args:
            inputSize: feature dimension

        """
        super(Nearest3DMemoryBGMixtueManager, self).__init__()

        self.bg_bank_size = bg_bank_size
        self.n_pos = num_pos
        self.num_orient = num_orient
        self.n_neg = num_noise  # num of sampled noise features from each image

        K = 1
        self.register_buffer('params', torch.tensor([K, T, -1, momentum]))
        stdv = 1.0 / math.sqrt(inputSize / 3)

        self.register_buffer(
            'memory', torch.rand(self.outputSize, inputSize).mul_(2 * stdv).add_(-stdv)
        )
        self.memory.requires_grad = False

        self.register_buffer(
            'accumulate_num',
            torch.zeros(self.n_pos, dtype=torch.long, device=self.memory.device),
        )

        self.accumulate_num.requires_grad = False
        self.feat_normalization = feat_normalization

        self.average_meter = VectorAverageMeter(num_pos * num_orient, inputSize)

    @property
    def fg_bank_size(self):
        return self.n_pos * self.num_orient

    @property
    def outputSize(self):
        return self.n_pos * self.num_orient + self.bg_bank_size

    def get_feature_banks(self):
        """
        returns:
        neural_features: V, C
        clutter_bank: NC, C
        """
        neural_features = self.memory[: self.fg_bank_size]
        neural_features = (
            neural_features.view(self.num_orient, self.n_pos, -1)
            .permute((1, 0, 2))
            .contiguous()
            .view(self.n_pos, -1)
        )

        clutter_bank = self.memory[self.fg_bank_size :]
        clutter_bank = F.normalize(clutter_bank, p=2, dim=1)
        return neural_features, clutter_bank

    def get_fg_feature_banks(self):
        """
        returns:
        neural_features: V, n_orient*C
        """
        neural_features = self.memory[: self.fg_bank_size]
        neural_features = (
            neural_features.view(self.num_orient, self.n_pos, -1)
            .permute((1, 0, 2))
            .contiguous()
            .view(self.n_pos, -1)
        )
        return neural_features

    def get_bg_feature_banks(self):
        """
        returns:
        clutter_bank: NC, C
        """
        clutter_bank = self.memory[self.fg_bank_size :]
        clutter_bank = F.normalize(clutter_bank, p=2, dim=1)
        return clutter_bank

    def set_bg_banks(self, centroids):
        self.memory[self.fg_bank_size :] = F.normalize(centroids, p=2, dim=1)

    # x: feature: [N, 128], y: indexes [N] -- a batch of data's index directly from the dataloader.
    def forward(self, x, y, visible):
        b, k, d_feat = x.shape

        momentum = self.params[3].item()

        distributed = torch.distributed.is_initialized()

        # x [n, k, d] * memory [l, d] = similarity : [n, k, l]
        if self.n_neg == 0:
            if self.feat_normalization:
                similarity = torch.matmul(x, torch.transpose(self.memory, 0, 1))
                noise_similarity = torch.zeros(1)
            else:
                similarity = -squared_euclidean_distance_matrix(
                    x.reshape(-1, d_feat), self.memory
                ).reshape(b, k, -1)
                noise_similarity = -1e5 * torch.ones(1)
        else:
            # t_ = x[:, 0:n_pos, :]
            if self.feat_normalization:
                similarity = torch.matmul(
                    x[:, 0 : self.n_pos, :], torch.transpose(self.memory, 0, 1)
                )
                noise_similarity = torch.matmul(
                    x[:, self.n_pos :, :], torch.transpose(self.memory[:, :], 0, 1)
                )
            else:
                similarity = -squared_euclidean_distance_matrix(
                    x[:, 0 : self.n_pos, :].reshape(-1, d_feat), self.memory
                ).reshape(b, self.n_pos, -1)
                noise_similarity = -squared_euclidean_distance_matrix(
                    x[:, self.n_pos :, :].reshape(-1, d_feat), self.memory[:]
                ).reshape(b, self.n_pos, -1)

        with torch.set_grad_enabled(False):
            # [n, k, k]

            y_onehot = one_hot(y, self.fg_bank_size).view(b, -1, self.fg_bank_size)
            y_idx = y.type(torch.long)

            # [B, n_neg, bg_bank_size]
            noise_mixture_similarity = torch.matmul(
                x[:, self.n_pos :, :],
                torch.transpose(self.memory[self.fg_bank_size :, :], 0, 1),
            )
            noise_idx = torch.argmax(noise_mixture_similarity, dim=2)
            noise_idx_onehot = one_hot(noise_idx, self.bg_bank_size).view(
                b, -1, self.bg_bank_size
            )
            count_noise = noise_idx_onehot.view(-1, self.bg_bank_size).sum(0)

            # update memory keypoints
            count = (y_onehot.view(-1, self.fg_bank_size) * visible.view(-1, 1)).sum(0)
            if distributed:
                dist.all_reduce(count, op=dist.ReduceOp.SUM)
                dist.all_reduce(count_noise, op=dist.ReduceOp.SUM)
            if (count > 0).any():
                if distributed:
                    x = allgather_tensor(x)
                    visible = allgather_tensor(visible)
                    y_onehot = allgather_tensor(y_onehot)
                    noise_idx_onehot = allgather_tensor(noise_idx_onehot)

                x_all = x[:, : self.n_pos, :].contiguous().view(-1, d_feat)

                x_mean = torch.matmul(
                    torch.transpose(y_onehot.view(-1, self.fg_bank_size), 0, 1),
                    x_all * visible.type(x.dtype).view(-1, 1),
                ) / (count.unsqueeze(1) + 1e-8)

                noise_mean = torch.matmul(
                    torch.transpose(noise_idx_onehot.view(-1, self.bg_bank_size), 0, 1),
                    x[:, self.n_pos :, :].contiguous().view(-1, d_feat),
                ) / (count_noise.unsqueeze(1) + 1e-8)

                mu = y_onehot.view(-1, self.fg_bank_size) @ x_mean
                m2 = y_onehot.view(-1, self.fg_bank_size).transpose(0, 1) @ (
                    (x_all - mu) ** 2 * visible.view(-1, 1)
                )
                self.average_meter.update_mean_var(
                    x_mean.detach(), m2.detach(), count.detach()
                )

                new_memory = torch.clone(self.memory)
                mem_ema = self.memory[: self.fg_bank_size][
                    count > 0
                ] * momentum + x_mean[count > 0] * (1 - momentum)

                noise_ema = self.memory[self.fg_bank_size :][
                    count_noise > 0
                ] * momentum + noise_mean[count_noise > 0] * (1 - momentum)

                new_memory[: self.fg_bank_size][count > 0] = (
                    F.normalize(mem_ema, dim=1, p=2)
                    if self.feat_normalization
                    else mem_ema
                )

                new_memory[self.fg_bank_size :][count_noise > 0] = (
                    F.normalize(noise_ema, dim=1, p=2)
                    if self.feat_normalization
                    else mem_ema
                )

                self.memory = new_memory

                self.accumulate_num += torch.sum(
                    (visible > 0)
                    .type(self.accumulate_num.dtype)
                    .to(self.accumulate_num.dtype),
                    dim=0,
                )

        # out.shape: [d, self.n_neg + self.n_pos]
        return similarity, y_idx, noise_similarity

    def set_zero(self, n_pos):
        self.accumulate_num = torch.zeros(
            n_pos, dtype=torch.long, device=self.memory.device
        )
        self.memory.fill_(0)

    def normalize_memory(self):
        self.memory.copy_(F.normalize(self.memory, p=2, dim=1))

    def cuda(self, device=None):
        super().cuda(device)
        self.accumulate_num = self.accumulate_num.cuda(device)
        self.memory = self.memory.cuda(device)
        return self
