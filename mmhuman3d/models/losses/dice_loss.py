import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weighted_loss


def dice_loss(pred, target, smooth = 1.):
    """
    Computes the Dice loss between predicted and target binary masks.
    """
    ph, pw = pred.size(1), pred.size(2)
    h, w = target.size(1), target.size(2)
    if ph != h or pw != w:
        pred = F.upsample(input=pred, size=(h, w), mode='bilinear', align_corners=True)
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return 1 - dice

@LOSSES.register_module()
class DICELoss(nn.Module):
    """DICELoss.

    Args:
        reduction (str, optional): The method that reduces the loss to a
            scalar. Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of the loss. Defaults to 1.0
    """

    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
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
        loss = loss_weight * dice_loss(pred, target)
        return loss

