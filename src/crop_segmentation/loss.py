import torch.nn as nn


# Classification loss. Binary CE or similar. See about class prevalence/weighting

# Challenge eval metric is mean IoU

class IoULoss(nn.Module):
    """
    Intersection over union (aka Jaccard) loss
    """

    def __init__(self):
        super(IoULoss, self).__init__()

    def forward(self, y, y_pred, e=1):
        # flatten label and prediction tensors
        y = y.view(-1)
        y_pred = y_pred.view(-1)

        intersection = (y * y_pred).sum()
        total = (y + y_pred).sum()
        union = total - intersection

        iou = (intersection + e) / (union + e)

        return 1 - iou


class DiceLoss(nn.Module):
    """
    Dice Similarity coefficient loss
    """

    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, y, y_pred):
        # flatten label and prediction tensors
        y = y.view(-1)
        y_pred = y_pred.view(-1)

        intersection = (y * y_pred).sum()

        dice = 2 * intersection / (y.sum() + y_pred.sum())

        return 1 - dice
