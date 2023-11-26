import numpy as np
import torch
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

        return -1 * dice


class MulticlassDiceLoss(nn.Module):
    """Reference: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch#Dice-Loss
    """

    def __init__(self, num_classes, softmax_dim=None):
        super().__init__()
        self.num_classes = num_classes
        self.softmax_dim = softmax_dim

    def forward(self, y_true, y_pred, smooth=1e-6):
        """The "reduction" argument is ignored. This method computes the dice
        loss for all classes and provides an overall weighted loss.
        """
        probabilities = y_pred.detach().clone()
        if self.softmax_dim is not None:
            probabilities = nn.Softmax(dim=self.softmax_dim)(y_pred)

        y_true_one_hot = nn.functional.one_hot(y_true.long(), num_classes=self.num_classes)
        y_true_one_hot = y_true_one_hot.squeeze().permute(0, 3, 1, 2)

        # HERE: the shape of target_one_hot and probabilities should be [batch_size, n_classes, img_size, img_size]

        # Ignore the background class
        probabilities = probabilities[:, 1:, :, :].squeeze()
        y_true_one_hot = y_true_one_hot[:, 1:, :, :].squeeze()

        # Multiply one-hot encoded ground truth labels with the probabilities to get the
        # predicted probability for the actual class.
        intersection = (y_true_one_hot * probabilities).sum()

        dice_coefficient = 2. * intersection / (probabilities.sum() + y_true_one_hot.sum() + smooth)
        dice_loss = -dice_coefficient  # Original impl. had a .log() here
        return dice_loss
