import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb
from ..builder import LOSSES
from .cross_entropy_loss import CrossEntropyLoss
from .utils import weighted_loss
from loguru import logger

def keypoints_to_pixel_index(keypoints, downsample_rate, original_img_size=(480, 640)):
    # original_img_size: (h, w)
    # line_size = 9
    line_size = original_img_size[1] // downsample_rate
    # round down, new coordinate (keypoints[:,:,0]//downsample_rate, keypoints[:, :, 1] // downsample_rate)
    return keypoints[:, :, 0] // downsample_rate * line_size + keypoints[:, :, 1] // downsample_rate

def ind_sel(target: torch.Tensor, ind: torch.Tensor, dim: int=1):
    """
    :param target: [... (can be k or 1), n > M, ...]
    :param ind: [... (k), M]
    :param dim: dim to apply index on
    :return: sel_target [... (k), M, ...]
    """
    assert len(ind.shape) > dim, "Index must have the target dim, but get dim: %d, ind shape: %s" % (dim, str(ind.shape))
    # ipdb.set_trace()
    target = target.expand(*tuple([ind.shape[k] if target.shape[k] == 1 else -1 for k in range(dim)] + [-1, ] * (len(target.shape) - dim)))

    ind_pad = ind

    if len(target.shape) > dim + 1:
        for _ in range(len(target.shape) - (dim + 1)):
            ind_pad = ind_pad.unsqueeze(-1)
        ind_pad = ind_pad.expand(*(-1, ) * (dim + 1), *target.shape[(dim + 1)::])

    return torch.gather(target, dim=dim, index=ind_pad)

def batched_index_select(t, dim, inds):
    dummy = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), t.size(2))
    out = t.gather(dim, dummy)  # b * e * f
    return out

def get_noise_pixel_index(keypoints, max_size, n_samples, bg_mask=None, dist_transform=None, T_dist=5):
    # keypoints [B, K] n is batch size, max_size is HxW
    n = keypoints.shape[0]
    # remove the point in keypoints by set probability to 0 otherwise 1 -> mask [n, size] with 0 or 1
    mask = torch.ones((n, max_size), dtype=torch.float32).to(keypoints.device)
    mask = mask.scatter(1, keypoints.type(torch.long), 0.) 
    if bg_mask is not None:
        mask *= bg_mask
    if dist_transform is not None:
        mask *= torch.exp(dist_transform / T_dist)
    # import ipdb; ipdb.set_trace()
    if torch.sum(mask.sum(dim=1) < 1e-6) > 0:
        # HACK (usually caused by inaccurate estimation of gt_cam_t)
        logger.info('torch.sum(mask.sum(dim=1) < 1e-6): '+str(torch.sum(mask.sum(dim=1) < 1e-6)))
        mask[mask.sum(dim=1) < 1e-6] = 1.0
    # generate the sample by the probabilities
    try:
        return torch.multinomial(mask, n_samples)
    except:
        logger.info('bug')
        logger.info('torch.sum(mask.sum(dim=1) < 1e-6): '+str(torch.sum(mask.sum(dim=1) < 1e-6)))
        logger.info('torch.isnan(mask).sum(): '+str(torch.isnan(mask).sum()))
        mask = torch.ones((n, max_size), dtype=torch.float32).to(keypoints.device)
        return torch.multinomial(mask, n_samples)

class GlobalLocalConverter(nn.Module):
    def __init__(self, local_size):
        super(GlobalLocalConverter, self).__init__()
        self.local_size = local_size
        self.padding = sum([[t - 1 - t // 2, t // 2] for t in local_size[::-1]], [])

    def forward(self, X):
        n, c, h, w = X.shape  # torch.Size([1, 2048, 8, 8])

        # N, C, H, W -> N, C, H + local_size0 - 1, W + local_size1 - 1
        X = F.pad(X, self.padding)

        # N, C, H + local_size0 - 1, W + local_size1 - 1 -> N, C * local_size0 * local_size1, H * W
        X = F.unfold(X, kernel_size=self.local_size)

        # N, C * local_size0 * local_size1, H * W -> N, C, local_size0, local_size1, H * W
        # X = X.view(n, c, *self.local_size, -1)

        # X:  N, C * local_size0 * local_size1, H * W
        return X

def sample_keypoint_features(X, keypoint_positions, map_shape, 
                                n_noise_points, noise_on_mask=False, 
                                feat_normalization=True, bg_mask=None,
                                dist_transform=None,
                                vert_coke_features=None,
                                iskpvisible=None):

    """
    Args:
    X (torch.Tensor): N x C x (HxW)
    keypoint_positions (torch.Tensor): N x K x 2 keypoint positions (r, c) downsampled
    bg_mask (torch.Tensor | None, optional): N x H x W object mask downsampled
    dist_transform (torch.Tensor | None, optional): N x H x W distance transform of 1-bg_mask 
    returns:
    sampled feature vectors N x (K+n_noise_points) x C
    """
    n = X.shape[0]

    # img_shape = X.shape[2:]
    net_out_dimension = X.shape[1]

    # keypoint_positioins are already downsampled
    keypoint_idx = keypoints_to_pixel_index(keypoints=keypoint_positions,
                                            downsample_rate=1,
                                            original_img_size=map_shape).type(torch.long)


    if n_noise_points == 0:
        keypoint_all = keypoint_idx
    else:
        if bg_mask is not None:
            # bg_mask = F.max_pool2d(bg_mask.unsqueeze(dim=1), kernel_size=net_stride, stride=net_stride, padding=(net_stride - 1) // 2)
            bg_mask = bg_mask.view(bg_mask.shape[0], -1)
            assert bg_mask.shape[1] == X.shape[2], 'mask_: ' + str(bg_mask.shape) + ' fearture_: ' + str(X.shape)
        if dist_transform is not None:
            # dist_transform = F.interpolate(dist_transform.unsqueeze(1), scale_factor=1/net_stride, recompute_scale_factor=False)
            dist_transform = dist_transform.view(dist_transform.shape[0], -1)
            assert dist_transform.shape[1] == X.shape[2], 'dist_transform_: ' + str(dist_transform.shape) + ' fearture_: ' + str(X.shape)
        if noise_on_mask:
            keypoint_noise = get_noise_pixel_index(keypoint_idx, max_size=X.shape[2], n_samples=n_noise_points, bg_mask=bg_mask, dist_transform=dist_transform)
        else:
            keypoint_noise = get_noise_pixel_index(keypoint_idx, max_size=X.shape[2], n_samples=n_noise_points, bg_mask=None)

        keypoint_all = torch.cat((keypoint_idx, keypoint_noise), dim=1)

    # N, C * local_size0 * local_size1, H * W -> N, H * W, C * local_size0 * local_size1
    X = torch.transpose(X, 1, 2)

    # N, H * W, C * local_size0 * local_size1 -> N, keypoint_all, C * local_size0 * local_size1
    # X = batched_index_select(X, dim=1, inds=keypoint_all)
    # vcf_org = ind_sel(X, dim=1, ind=keypoint_idx)
    if vert_coke_features is not None: # and random.random() > 0.5:
        X = ind_sel(X, dim=1, ind=keypoint_noise) 
        # import ipdb; ipdb.set_trace()
        # X = torch.ones_like(X)
        X = torch.cat((vert_coke_features, X), dim=1) #vert_coke_features
    else:
        X = ind_sel(X, dim=1, ind=keypoint_all) 

    X = F.normalize(X, p=2, dim=2) if feat_normalization else X 
    X = X.view(n, -1, net_out_dimension)

    return X

def mask_remove_near(keypoints, thr, dtype_template=None, num_neg=0, neg_weight=1, eps=1e5, n_orient=1):
    '''
    neg_weight: weight for clutter_similarity neg_weight*exp(f_i^T f_c)
    '''
    if dtype_template is None:
        dtype_template = torch.ones(1, dtype=torch.float32)
    # keypoints -> [n, k, 2]
    with torch.no_grad():
        # distance -> [n, k, k]
        distance = torch.sum((torch.unsqueeze(keypoints, dim=1) - torch.unsqueeze(keypoints, dim=2)).pow(2), dim=3).pow(0.5)
        if num_neg == 0:
            return ((distance <= thr).type_as(dtype_template) - torch.eye(keypoints.shape[1]).type_as(dtype_template).unsqueeze(dim=0)) * eps
        else:
            tem = (distance <= thr).type_as(dtype_template) - torch.eye(keypoints.shape[1]).type_as(dtype_template).unsqueeze(dim=0)
            return torch.cat([tem.repeat((1,1,n_orient)) * eps, - torch.ones(keypoints.shape[0: 2] + (num_neg, )).type_as(dtype_template) * math.log(neg_weight)], dim=2)

def mask_remove_adjacent(adj_mat, n_dist=1, dtype_template=None, num_neg=0, neg_weight=1, eps=1e5, n_orient=1):
    '''
    adj_mat: adjacency matrix of the mesh
    n_dist: consider the n-hop neighbor
    '''
    if dtype_template is None:
        dtype_template = torch.ones(1, dtype=torch.float32)
    # adj_mat -> [1, k, 2]
    # if n_dist > 1:
    #     for i in range(n_dist):
    #         adj_mat = (torch.bmm(adj_mat, adj_mat) > 0).long()

    with torch.no_grad():
        if num_neg == 0:
            return (adj_mat.type_as(dtype_template) * (1 - torch.eye(adj_mat.shape[1]).type_as(dtype_template).unsqueeze(dim=0))) * eps
        else:
            tem = adj_mat.type_as(dtype_template) * (1 - torch.eye(adj_mat.shape[1]).type_as(dtype_template).unsqueeze(dim=0))
            return torch.cat([tem.repeat((1,1,n_orient)) * eps, - torch.ones(adj_mat.shape[0: 2] + (num_neg, )).type_as(dtype_template) * math.log(neg_weight)], dim=2)

def remove_outside(kp, img_size, vis):
    """
    kp: row, col
    img_size: h, w
    """
    invisible = torch.logical_or(
                torch.logical_or(
                torch.logical_or(kp[:, :, 0] < 0, 
                                kp[:, :, 1] < 0), 
                                kp[:, :, 0] > img_size[0]-1), 
                                kp[:, :, 1] > img_size[1]-1)
    vis = vis * (~invisible)
    kp[invisible] = 0 # remove this?
    return kp, vis


@LOSSES.register_module()
class CoKeLossVoGE(nn.Module):
    """CrossEntropyLoss.

    Args:
        reduction (str, optional): The method that reduces the loss to a
            scalar. Options are "none", "mean" and "sum".
        loss_contrastive_weight (float, optional): The weight of the loss. Defaults to 1.0
    """

    def __init__(self, n_noise_points, num_neg,
                noise_on_mask=True,
                weight_noise=5e-3, n_orient=3, 
                feat_normalization=True,
                local_size=1, T=0.07,
                loss_contrastive_weight=1.0,
                loss_noise_reg_weight=1.0,
                ):
        super().__init__()
        
        self.n_noise_points = n_noise_points
        self.num_neg = num_neg
        self.noise_on_mask = noise_on_mask
        self.weight_noise = weight_noise
        self.n_orient = n_orient
        self.feat_normalization = feat_normalization
        self.T = T
        self.local_size = [local_size, local_size]
        self.loss_contrastive_weight = loss_contrastive_weight
        self.loss_noise_reg_weight =loss_noise_reg_weight 

        self.ce_loss = CrossEntropyLoss()
        self.converter = GlobalLocalConverter(self.local_size)

    def forward(self,
                coke_features,
                keypoint_positions,
                has_smpl,
                iskpvisible,
                feature_bank,
                adj_mat,
                vert_orients=None,
                bg_mask=None,
                weight=None,
                vert_coke_features=None
                ):
        """Forward function of loss.

        Args:
            coke_features (torch.Tensor): N x C x H x W, CoKe feature maps.
            keypoint_positions (torch.Tensor): N x K x 2, (row, col) vertex positions in 2D.
            iskpvisible (torch.Tensor): N x K, vertex visibility
            feature_bank (nn.Module): feature bank that stores the neural features
            vert_orients (torch.Tensor): N x K, orientation of the vertices
            weight (torch.Tensor, optional): Weight of the loss for each
                prediction. Defaults to None.
        Returns:
            torch.Tensor: The calculated loss
        """
        keypoint_positions, iskpvisible = remove_outside(keypoint_positions, 
                                                        bg_mask.shape[1:], # coke_features.shape[2:], 
                                                        iskpvisible)
        iskpvisible[has_smpl == 0] = 0
        # N, C * local_size0 * local_size1, H * W
        bs, y_num, _ = keypoint_positions.shape
        index = torch.Tensor([[k for k in range(y_num)]] * bs).to(coke_features.device)
        if self.n_orient > 1:
            index += vert_orients * y_num
        index = index.to(coke_features.device)
        X = self.converter(coke_features)
        keypoint_coke_features = sample_keypoint_features(X, keypoint_positions, 
                                                map_shape=coke_features.shape[2:],
                                                n_noise_points=self.n_noise_points,
                                                noise_on_mask=self.noise_on_mask,
                                                feat_normalization=self.feat_normalization,
                                                bg_mask=bg_mask,
                                                vert_coke_features=vert_coke_features,
                                                iskpvisible=iskpvisible,
                                                )
        # keypoint_coke_features = coke_features

        # get B, K, (n_vert*n_orient+num_neg)
        get, y_idx, noise_sim = feature_bank(keypoint_coke_features, index, iskpvisible) 
        get /= self.T

        mask_distance_legal = mask_remove_adjacent(adj_mat, n_dist=3, num_neg=self.num_neg,
                                            dtype_template=get, 
                                            neg_weight=self.weight_noise, 
                                            n_orient=self.n_orient)

        loss_contrastive = self.ce_loss(
                            ((get - mask_distance_legal)[has_smpl == 1].view(-1, get.shape[2]))[iskpvisible[has_smpl == 1].view(-1), :],
                            y_idx[has_smpl == 1].view(-1)[iskpvisible[has_smpl == 1].view(-1)])

        loss_contrastive = torch.mean(loss_contrastive) * self.loss_contrastive_weight

        if self.n_noise_points > 0:
            if self.feat_normalization:
                loss_noise_reg = torch.mean(noise_sim[has_smpl == 1])
            else:
                loss_noise_reg = torch.exp(torch.mean(noise_sim[has_smpl == 1]))
        else:
            loss_noise_reg = torch.zeros(1).to(coke_features.device)

        loss_noise_reg *= self.loss_noise_reg_weight
        return loss_contrastive, loss_noise_reg