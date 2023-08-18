import numpy as np
import torch
import torch.nn.functional as F
from mmhuman3d.core.conventions.constants import vertex_to_part, nemo_part_list, openpose_links
import ipdb

def compute_orient_weights(cosx, N, theta):
    """
    Arguments:
        cosx: B, P
        N: Number of orientations
        theta: threshold
    returns:
        orient_weights: B, N
    """
    
    x = torch.arccos(cosx)

    orient_weights = torch.stack([  (torch.sigmoid(theta*(x - k * np.pi / N)) + 
                                    torch.sigmoid(-theta*(x - (k+1) * np.pi / N)) - 1) 
                                                        for k in range(N)], dim=2)
    return orient_weights


def get_vert_orient_weights(
                openpose_joints,
                cameras,
                n_orient=2,
                theta=10
                ):
    """
    openpose_joints: B, J, 3
    returns: 
        vert_orient_weights: B, V, 3
    """
    
    part_orients = []
    for part in nemo_part_list:
        lnk = openpose_links[part]
        part_vec = openpose_joints[:, lnk[1]] - openpose_joints[:, lnk[0]]
        part_orients.append(part_vec.unsqueeze(1))

    part_orients = torch.cat(tuple(part_orients), dim=1)
    part_orients = F.normalize(torch.einsum('ijk,ilk->ilj', cameras.R.transpose(1, 2).float(), part_orients), p=2, dim=2)
    z_vec = torch.tensor([[0, 0, 1]], dtype=torch.float32).to(part_orients.device)
    part_proj = torch.bmm(part_orients, z_vec.unsqueeze(2).expand(part_orients.shape[0], -1, -1)).squeeze(-1) # N, P
    
    part_orients_label = torch.zeros_like(part_proj)

    if n_orient == 3:
        thr = 0.3 #0.01 #0.3
        orient_weights = torch.stack([
                                torch.sigmoid(theta*(part_proj+thr))+torch.sigmoid(theta*(-part_proj+thr))-1,
                                torch.sigmoid(theta*(part_proj-thr)),
                                torch.sigmoid(theta*(-part_proj-thr))], dim=2)
    elif n_orient == 2:
        orient_weights = torch.stack([
                                torch.sigmoid(theta*(part_proj)),
                                torch.sigmoid(theta*(-part_proj))], dim=2)
    elif n_orient > 3:
        orient_weights = compute_orient_weights(part_proj, n_orient, theta)
    else:
        orient_weights = torch.ones_like(part_proj)[:, :, None]
    # orient_weights: B, P, N_orient

    v2p = torch.tensor(vertex_to_part, device=part_orients_label.device) # [V]
    
    vert_orient_weights = orient_weights.gather(1, v2p[None, :, None].expand(orient_weights.shape[0], -1, n_orient))
    return vert_orient_weights

def get_detected_2d_vertices(predicted_map, bbox_xyxy, feature_bank, downsample_rate, n_orient=None, obj_mask=None ):
    """
    Object detected vertices in 2D
    :predicted_map: B, C, H, W
    :bbox_xyxy: human bbox_xyxy
    :feature_bank: n_vert, n_orient*C
    
    returns:
    :max_idx: [B, K, 2], (x, y)
    """
    predicted_map = F.normalize(predicted_map, p=2, dim=1)
    feature_bank_ = feature_bank.view(feature_bank.shape[0]*n_orient, -1) # nv, no*c -> nv*no, c
    hmap = F.conv2d(predicted_map, feature_bank_.unsqueeze(2).unsqueeze(3)) # B, nv*no, H, W
    if n_orient is not None:
        hmap = hmap.view(hmap.shape[0], -1, n_orient, hmap.shape[2], hmap.shape[3]).max(dim=2)[0]

    # Get argmax detection inside mask

    # obj_mask = obj_mask.to(device)
    # stride_ = downsample_rate
    # obj_mask = F.max_pool2d(obj_mask.unsqueeze(dim=1),
    #                         kernel_size=stride_,
    #                         stride=stride_,
    #                         padding=(stride_ - 1) // 2) # why padding?
    # hmap = hmap * obj_mask
    B, _, H, W = hmap.shape
    bbox_xyxy = bbox_xyxy / downsample_rate
    ys, xs = torch.meshgrid(torch.arange(H, device=hmap.device), 
                            torch.arange(W, device=hmap.device))
    mask = torch.logical_and(
        torch.logical_and(xs[None] > bbox_xyxy[:, 0][:, None, None], xs[None] < bbox_xyxy[:, 2][:, None, None]), 
        torch.logical_and(ys[None] > bbox_xyxy[:, 1][:, None, None], ys[None] < bbox_xyxy[:, 3][:, None, None])
    ).unsqueeze(1) # B, 1, H, W

    # mask = torch.zeros_like(hmap)
    # mask[:, :, int(bbox[2]):int(bbox[3])+1, int(bbox[0]):int(bbox[1])+1] = 1
    hmap = hmap * mask

    # [n, k, h, w]
    w = hmap.size(3)
    hmap = hmap.view(*hmap.shape[0:2], -1)

    conf, max_ = torch.max(hmap, dim=2)
    max_idx = torch.zeros((*hmap.shape[0:2], 2),
                        dtype=torch.long).to(hmap.device)

    max_idx[:, :, 0] = max_ % w
    max_idx[:, :, 1] = max_ // w
    
    # vertices2d_det = torch.cat((max_idx, conf.unsqueeze(2)), dim=2)
    # return vertices2d_det
    # # max_idx = max_idx * stride_ + stride_ // 2
    return max_idx, conf