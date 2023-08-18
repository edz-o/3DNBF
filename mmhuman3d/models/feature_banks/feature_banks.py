import torch
from torch.autograd import Function
import torch.nn.functional as F
import torch.distributed as dist
from torch import nn
import math
import ipdb
from mmcv.runner.base_module import BaseModule
from ..builder import FEATURE_BANKS
from mmhuman3d.utils.dist_utils import allgather_tensor
from mmhuman3d.utils.image_utils import one_hot
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


class AverageMeter:
    """Welford, https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance"""

    def __init__(self):
        self.count = 0
        self.mean = 0
        self.m2 = 0  # \sum (x_i-\bat{x}_n)^2

    def update_mean_var(self, new_value):
        self.count += 1
        delta = new_value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (new_value - self.mean)

    def get_mean(self):
        return self.mean

    def get_var(self):
        return self.m2 / self.count if self.count > 1 else 0

    def get_sample_var(self):
        return self.m2 / (self.count - 1) if self.count > 1 else 0


class VectorAverageMeter:
    """Welford, https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    (TODO) numerical overflow?
    """

    def __init__(self, N, C, device='cpu'):
        self.count = torch.zeros(N, device=device)  # N
        self.mean = torch.zeros(N, C, device=device)  # N, C
        self.m2 = torch.zeros(N, C, device=device)  # N, C, \sum (x_i-\bat{x}_n)^2

    def to(self, device):
        self.count = self.count.to(device)
        self.mean = self.mean.to(device)
        self.m2 = self.m2.to(device)

    def update_mean_var(self, batch_mean, batch_m2, batch_count):
        if self.count.device != batch_count.device:
            self.to(batch_count.device)
        new_mean = self.count[:, None] * self.mean + batch_count[:, None] * batch_mean
        new_count = self.count + batch_count
        new_mean[new_count > 1] = (
            new_mean[new_count > 1] / new_count[new_count > 1, None]
        )

        self.m2 = (
            self.m2
            + batch_m2
            + self.count[:, None] * (self.mean - new_mean) ** 2
            + batch_count[:, None] * (batch_mean - new_mean) ** 2
        )
        self.count = new_count
        self.mean = new_mean

    def get_count(self):
        return self.count

    def get_mean(self):
        return self.mean

    def get_var(self):
        var = self.m2
        var[self.count > 1] = (
            var[self.count > 1] / (self.count[self.count > 1])[:, None]
        )
        return var

    def get_sample_var(self):
        var = self.m2
        var[self.count > 1] = (
            var[self.count > 1] / (self.count[self.count > 1] - 1)[:, None]
        )
        return var


@FEATURE_BANKS.register_module()
class Nearest3DMemoryManager(BaseModule):
    def __init__(
        self,
        inputSize,
        outputSize,
        K,
        num_pos,
        num_orient,
        T=0.07,
        momentum=0.5,
        Z=None,
        max_groups=-1,
        num_noise=-1,
        feat_normalization=False,
    ):
        super(Nearest3DMemoryManager, self).__init__()
        self.nLem = outputSize
        self.K = K
        self.num_pos = num_pos
        self.num_orient = num_orient

        self.register_buffer('params', torch.tensor([K, T, -1, momentum]))
        stdv = 1.0 / math.sqrt(inputSize / 3 - 1 / 15)

        self.register_buffer(
            'memory', torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv)
        )
        # self.memory = torch.rand(outputSize, inputSize).mul_(2 * stdv).add_(-stdv)
        self.memory.requires_grad = False

        self.lru = 0
        if max_groups > 0:
            self.max_lru = max_groups
        else:
            self.max_lru = -1

        if num_noise < 0:
            self.num_noise = self.K
        else:
            self.num_noise = num_noise  # num of sampled noise features from each image

        self.register_buffer(
            'accumulate_num',
            torch.zeros(self.num_pos, dtype=torch.long, device=self.memory.device),
        )

        self.accumulate_num.requires_grad = False
        self.feat_normalization = feat_normalization

        self.average_meter = VectorAverageMeter(num_pos * num_orient, inputSize)

    @property
    def fg_bank_size(self):
        return self.num_pos * self.num_orient

    @property
    def outputSize(self):
        return self.n_pos * self.num_orient + self.bg_bank_size

    def get_feature_banks(self):
        """
        returns:
        neural_features: V, C
        clutter_bank: 1, C
        """
        neural_features = self.memory[: self.fg_bank_size]
        neural_features = (
            neural_features.view(self.num_orient, self.num_pos, -1)
            .permute((1, 0, 2))
            .contiguous()
            .view(self.num_pos, -1)
        )

        clutter_bank = self.memory[self.fg_bank_size :]
        clutter_bank = F.normalize(
            torch.mean(clutter_bank, dim=0, keepdim=True), p=2, dim=1
        )
        return neural_features, clutter_bank

    def get_feature_banks_original_order(self):
        """
        returns:
        neural_features: n_orient, V, C
        clutter_bank: 1, C
        """
        neural_features = self.memory[: self.fg_bank_size]
        neural_features = neural_features.view(self.num_orient, self.num_pos, -1)

        clutter_bank = self.memory[self.fg_bank_size :]
        clutter_bank = F.normalize(
            torch.mean(clutter_bank, dim=0, keepdim=True), p=2, dim=1
        )
        return neural_features, clutter_bank

    def get_fg_feature_banks(self):
        """
        returns:
        neural_features: V, n_orient*C
        """
        neural_features = self.memory[: self.fg_bank_size]
        neural_features = (
            neural_features.view(self.num_orient, self.num_pos, -1)
            .permute((1, 0, 2))
            .contiguous()
            .view(self.num_pos, -1)
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

    def forward(self, x, y, visible):
        n_pos = self.num_pos
        n_neg = self.num_noise
        b, k, d_feat = x.shape
        group_size = int(self.params[0].item())
        momentum = self.params[3].item()

        assert group_size == 1, 'Currently only support group size = 1'
        n_class = self.num_orient * n_pos // group_size

        distributed = torch.distributed.is_initialized()

        if (
            self.max_lru == -1
            and n_neg > 0
            and x.shape[0] <= (self.nLem - n_pos * self.num_orient) / n_neg
        ):
            if distributed:
                self.max_lru = (self.memory.shape[0] - n_pos * self.num_orient) // (
                    n_neg * x.shape[0] * dist.get_world_size()
                )
            else:
                self.max_lru = (self.memory.shape[0] - n_pos * self.num_orient) // (
                    n_neg * x.shape[0]
                )

        # x [n, k, d] * memory [l, d] = similarity : [n, k, l]
        if n_neg == 0:
            if self.feat_normalization:
                similarity = torch.matmul(x, torch.transpose(self.memory, 0, 1))
                noise_similarity = torch.zeros(1)
            else:
                similarity = -squared_euclidean_distance_matrix(
                    x.reshape(-1, d_feat), self.memory
                ).reshape(b, k, -1)
                noise_similarity = -1e5 * torch.ones(1)
        else:
            if self.feat_normalization:
                similarity = torch.matmul(
                    x[:, 0:n_pos, :], torch.transpose(self.memory, 0, 1)
                )
                noise_similarity = torch.matmul(
                    x[:, n_pos:, :],
                    torch.transpose(self.memory[0 : n_pos * self.num_orient, :], 0, 1),
                )
            else:
                similarity = -squared_euclidean_distance_matrix(
                    x[:, 0:n_pos, :].reshape(-1, d_feat), self.memory
                ).reshape(b, n_pos, -1)
                noise_similarity = -squared_euclidean_distance_matrix(
                    x[:, n_pos:, :].reshape(-1, d_feat),
                    self.memory[0 : n_pos * self.num_orient],
                ).reshape(b, n_pos, -1)

        with torch.set_grad_enabled(False):
            y_onehot = one_hot(y, n_class).view(x.shape[0], -1, n_class)

            y_idx = y.type(torch.long)

            # update memory keypoints
            count = (y_onehot.view(-1, n_class) * visible.view(-1, 1)).sum(0)
            if distributed:
                dist.all_reduce(count, op=dist.ReduceOp.SUM)
            if (count > 0).any():
                if distributed:
                    x = allgather_tensor(x)
                    visible = allgather_tensor(visible)
                    y_onehot = allgather_tensor(y_onehot)

                x_all = x[:, :n_pos, :].contiguous().view(-1, d_feat)

                x_mean = torch.matmul(
                    torch.transpose(y_onehot.view(-1, n_class), 0, 1),
                    x_all * visible.type(x.dtype).view(-1, 1),
                ) / (count.unsqueeze(1) + 1e-8)

                mu = y_onehot.view(-1, n_class) @ x_mean
                m2 = y_onehot.view(-1, n_class).transpose(0, 1) @ (
                    (x_all - mu) ** 2 * visible.view(-1, 1)
                )
                self.average_meter.update_mean_var(
                    x_mean.detach(), m2.detach(), count.detach()
                )

                new_memory = torch.clone(self.memory)
                mem_ema = self.memory[:n_class][count > 0] * momentum + x_mean[
                    count > 0
                ] * (1 - momentum)
                new_memory[:n_class][count > 0] = (
                    F.normalize(mem_ema, dim=1, p=2)
                    if self.feat_normalization
                    else mem_ema
                )
                if n_neg > 0:
                    if x.shape[0] > (self.nLem - n_class) / n_neg:
                        # enough negative sampling
                        mem_ema = (
                            x[:, n_pos::, :]
                            .contiguous()
                            .view(-1, x.shape[2])[0 : self.memory.shape[0] - n_class]
                        )
                        new_memory[n_class:] = (
                            F.normalize(mem_ema, dim=1, p=2)
                            if self.feat_normalization
                            else mem_ema
                        )
                    else:
                        mem_ema = (
                            x[:, n_pos::, :].contiguous().view(-1, x.shape[2])
                        )  # (B x n_neg), C
                        new_memory[
                            n_class
                            + self.lru * n_neg * x.shape[0] : n_class
                            + (self.lru + 1) * n_neg * x.shape[0]
                        ] = (
                            F.normalize(mem_ema, dim=1, p=2)
                            if self.feat_normalization
                            else mem_ema
                        )
                        self.lru += 1
                        self.lru = self.lru % self.max_lru
                self.memory = new_memory
                self.accumulate_num += torch.sum(
                    (visible > 0)
                    .type(self.accumulate_num.dtype)
                    .to(self.accumulate_num.dtype),
                    dim=0,
                )

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
