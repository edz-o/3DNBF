import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weighted_loss


def mask_alignment_loss(vert2d, mask):
    """
    Computes the Dice loss between predicted and target binary masks.
    """
    B, H, W = mask.shape
    vert2d = vert2d / torch.tensor([W, H], device=vert2d.device)[None, None, :]
    loss = 0
    for i in range(B):
        points = torch.where(mask[i]>0)
        mask_points = torch.cat((points[1][:, None], points[0][:, None]), dim=1) # M x 2
        mask_points = mask_points.float() / torch.tensor([W, H], device=vert2d.device)[None, :]
        dist_mat = torch.norm(vert2d[i, :, None] - mask_points[None, :], dim=2) # V x M
        loss += dist_mat.min(dim=0)[0].sum() + dist_mat.min(dim=1)[0].sum()
    return loss

@LOSSES.register_module()
class MaskAlignmentLoss(nn.Module):
    """Mask Alignment Loss.

    Args:
        reduction (str, optional): The method that reduces the loss to a
            scalar. Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of the loss. Defaults to 1.0
    """

    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self,
                vert2d,
                mask,
                weight=None,
                avg_factor=None,
                loss_weight_override=None):
        """Forward function of loss.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): Weight of the loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None.
        Returns:
            torch.Tensor: The calculated loss
        """
        loss_weight = (
            loss_weight_override
            if loss_weight_override is not None else self.loss_weight)
        loss = loss_weight * mask_alignment_loss(vert2d, mask)
        return loss

