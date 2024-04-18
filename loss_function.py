import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean', gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, inputs, target):
        log_p = self.ce(inputs, target)
        p = torch.exp(-log_p)
        loss = (1 - p) ** self.gamma * log_p
        return loss.mean()
