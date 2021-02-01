import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.ignore_index = -100

    def forward(self, logits, labels, mask, eps=1e-8):
        """Computes the Sørensen–Dice loss.
        Note that PyTorch optimizers minimize a loss. In this
        case, we would like to maximize the dice loss so we
        return the negated dice loss.
        Args:
            labels: a tensor of shape [B].
            logits: a tensor of shape [B, C]. Corresponds to
                the raw output or logits of the model.
            eps: added to the denominator for numerical stability.
        Returns:
            dice_loss: the Sørensen–Dice loss.
        """
        num_classes = logits.shape[1]
        labels[labels == self.ignore_index] = 0
        labels_1_hot = torch.eye(num_classes)[labels]
        prob = F.softmax(logits, dim=1)
        labels_1_hot = labels_1_hot.type(logits.type())
        mod_mask = mask.unsqueeze(-1).repeat(1, num_classes)
        intersection = torch.sum(prob * labels_1_hot * mod_mask, dim=-1)
        cardinality = torch.sum(prob + labels_1_hot * mod_mask, dim=-1)
        dice_loss = (2.0 * (intersection + eps) / (cardinality + eps)).mean()
        return 1.0 - dice_loss
