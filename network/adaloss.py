import torch
from torch import nn


class AdaLoss(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, gamma=0):
        super(AdaLoss, self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, labels):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            labels: ground truth labels with shape (num_classes)
        """
        num = len(inputs)
        log_pts = []
        pts = []
        for i in range(num):
            log_pt = self.logsoftmax(inputs[i])
            log_pts.append(log_pt)
            pts.append(log_pt.data.exp())
        targets = torch.zeros(log_pts[0].size(), requires_grad=False).scatter_(1, labels.unsqueeze(1).data.cpu(), 1)
        targets = targets.to(inputs[0].device)
        
        for i in range(num):
            if i == 0:
                loss = -(targets * log_pts[i]).mean(0).sum()
            if i > 0:
                if self.gamma == 0:
                    loss += -(targets * log_pts[i]).mean(0).sum()
                else:
                    indexs = torch.repeat_interleave(labels.unsqueeze(1), inputs[0].shape[1], dim=1)
                    predict = torch.gather(pts[i-1], 1, indexs)
                    ada_weight = (1 - predict).pow(self.gamma)
                    ada_weight = ada_weight / ada_weight.mean(dim=0, keepdim=True)
                    loss += -(ada_weight * targets * log_pts[i]).mean(0).sum()
        return loss

