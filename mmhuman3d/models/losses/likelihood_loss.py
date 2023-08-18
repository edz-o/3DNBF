import torch
import torch.nn as nn
import numpy as np

from mmhuman3d.utils.neuralsmpl_utils import get_vert_orient_weights
from .utils import weighted_loss
from ..builder import LOSSES

def soft_maximum(a, b, t1=1, t2=1):
    """ soft version of element-wise maximum
    Args:
        a, b: (B, ...)
    Returns:
        x: (B, ...)
    """
    x = torch.cat((a.unsqueeze(1), b.unsqueeze(1)), dim=1)
    t = torch.tensor([t1, t2], dtype=x.dtype, device=x.device)
    return (x * torch.softmax(x/t.view(1, -1, *[1,]*(a.dim()-1)), dim=1)).sum(dim=1)

def soft_maximum_old(a, b, t1=1, t2=1):
    return (((a)**2 + (b)**2)/(a + b + 1e-9))

def soft_loss_fun(obj_s, clu_s, weights=None, reduce='mean', normalize=True, mask=None, t_obj=1/10, t_clu=1/10):
    """
    Args:
        obj_s: (B, H, W)
        clu_s: (B, H, W)
        normalize: whether normalize obj_s and clu_s to [0,1]
        mask: (B, H, W), only compute loss in mask
    Returns:
        rasterization_loss: 
    """
    if normalize:
        obj_s = torch.clamp(0.5 * obj_s + 0.5, 1e-8, 1) 
        clu_s = (0.5 * clu_s + 0.5) #* 0.5

    if reduce == 'mean':
        l = 1 - soft_maximum(obj_s, clu_s, t_obj, t_clu)
        if mask is not None:
            return (l * mask).sum(dim=[1, 2]) / (mask.sum(dim=[1, 2]) + 1e-9)
        else:
            return l.mean(dim=[1, 2]) 
    elif reduce is None:
        return 1 - soft_maximum(obj_s, clu_s, t_obj, t_clu) 

@weighted_loss
def likelihood_loss(projected_map, predicted_map, clutter_scores, mask, 
                        pixel_orient_weights, n_orient, pixel_var, options, 
                        ):
    """
    Args:
        predicted_map: B, C, H, W
        projected_map: B, 
        clutter_scores: B, H, W
        mask: B, H, W
        pixel_orient_weights: B, N_orient, H, W
        options: hyperparamters
    Returns:
        rasterization_loss: (B,) or (B, H, W)
    """
    B, _, H, W = projected_map.shape 
    projected_map = projected_map.view(B, n_orient, -1, H, W) # B, N_orient, C, H, W
    projected_map = torch.nn.functional.normalize(projected_map, dim=2, p=2) 

    object_score = torch.sum(projected_map * predicted_map.unsqueeze(1), dim=2) # B, N_orient, H, W
    
    options.bUseVariance = False
    if pixel_var is not None:
        # pixel_var: B, N_orient, H, W
        pixel_var += (1 - mask) * 1e8
    # import pdb; pdb.set_trace()
    if n_orient > 1:
        if options.bUseVariance:
            object_score = (pixel_orient_weights * object_score / (pixel_var) ).sum(1)
        else:
            object_score = (pixel_orient_weights * object_score ).sum(1)
        # object_score: B, H, W
    else:
        if options.bUseVariance:
            object_score = object_score / pixel_var
        else:
            object_score = object_score

    # object_score = -torch.ones_like(object_score) * mask + object_score * (1 - mask)

    # Rasterization loss
    if options.bUseVariance:
        clutter_scores = clutter_scores / max(pixel_var.min(dim=1)[0][mask>0].min(), 1e-8) # approximate clutter variance
    else:
        clutter_scores = clutter_scores

    
    if options.disable_occ:
        # disable occlusion model
        heatmap = (object_score * mask + clutter_scores * (1 - mask))
        # heatmap = object_score * mask
        rasterization_loss = 1 - heatmap.mean(dim=[1,2])
    else:
        heatmap = torch.maximum(clutter_scores, object_score)
        rasterization_loss = soft_loss_fun(object_score, clutter_scores, normalize=True)
    return rasterization_loss

@LOSSES.register_module()
class LikelihoodLoss(nn.Module):
    """LikelihoodLoss for NeuralSMPL.

    Args:
        reduction (str, optional): The method that reduces the loss to a
            scalar. Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of the loss. Defaults to 1.0.
    """

    def __init__(self, reduction='mean', loss_weight=1.0):
        super().__init__()
        assert reduction in (None, 'none', 'mean', 'sum')
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred, target, clutter_scores, mask, 
                pixel_orient_weights, n_orient, pixel_var, options, 
                loss_weight_override=None,
                reduction_override=None):
        """Forward function of loss.

        Args:
            pred (torch.Tensor): The prediction. Shape should be (N, K, 2/3)
                B: batch size. K: number of keypoints.
            target (torch.Tensor): The learning target of the prediction.
                Shape should be the same as pred.
            pred_conf (optional, torch.Tensor): Confidence of
                predicted keypoints. Shape should be (N, K).
            
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            loss_weight_override (float, optional): The overall weight of loss
                used to override the original weight of loss.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_weight = (
            loss_weight_override
            if loss_weight_override is not None else self.loss_weight)

        loss = loss_weight * likelihood_loss(pred, target, 
                        clutter_scores=clutter_scores, 
                        mask=mask, 
                        pixel_orient_weights=pixel_orient_weights, 
                        n_orient=n_orient, 
                        pixel_var=pixel_var, 
                        options=options, 
                        reduction=reduction,
                        )
        return loss