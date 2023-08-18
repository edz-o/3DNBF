import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weighted_loss

def batch_cov(points):
    B, N, D = points.size()
    mean = points.mean(dim=1).unsqueeze(1)
    diffs = (points - mean).reshape(B * N, D)
    prods = torch.bmm(diffs.unsqueeze(2), diffs.unsqueeze(1)).reshape(B, N, D, D)
    bcov = prods.sum(dim=1) / (N - 1)  # Unbiased estimate
    return bcov  # (B, D, D)

def corr_loss(pred, target, smooth = 1.):
    """
    Computes the Pearson Correlation Coefficient loss between predicted and target sequence.
    Args:
        pred (torch.Tensor): Predicted sequence. Shape: (N, L)
        target (torch.Tensor): Target sequence. Shape: (N, L)
    Returns:
        torch.Tensor: Loss value.
    """

    corr = batch_cov(torch.cat((pred[:,:,None], target[:,:,None]), dim=2))[:, 0, 1] / (torch.std(pred, dim=1) * torch.std(target, dim=1))
    return -corr

@LOSSES.register_module()
class CorrLoss(nn.Module):
    """CorrLoss.

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
        loss = loss_weight * corr_loss(pred, target)
        return loss

