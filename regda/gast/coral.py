"""
@Project : rsda
@File    : coral.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2023/3/31 下午3:12
@e-mail  : 1183862787@qq.com
"""
# https://arxiv.org/pdf/1607.01719.pdf
import torch
import torch.nn as nn
import torch.nn.functional as nnf


class CoralLoss(nn.Module):

    def __init__(self, is_sqrt=False):
        """
        Coral loss implement for eq(1) in https://arxiv.org/pdf/1607.01719.pdf
        Args:
            is_sqrt:
        """
        super().__init__()
        self.is_sqrt = is_sqrt

    def forward(self, source, target):
        """
        Args:
            source: tensor with shape=(instance num, feature dimension)
            target: tensor with shape=(instance num, feature dimension)
        Returns:
            deep coral loss
        """
        d = source.data.shape[1]
        ns, nt = source.data.shape[0], target.data.shape[0]
        # source covariance
        xm = torch.mean(source, 0, keepdim=True) - source
        xc = xm.t() @ xm / (ns - 1)

        # target covariance
        xmt = torch.mean(target, 0, keepdim=True) - target
        xct = xmt.t() @ xmt / (nt - 1)

        # frobenius norm between source and target
        loss = torch.sum(torch.mul((xc - xct), (xc - xct)))
        loss = (loss.sqrt() if self.is_sqrt else loss) / (4 * d * d)
        return loss


class CoralLoss2(nn.Module):

    def __init__(self, is_sqrt=False):
        """
        Coral loss implement for eq(1, 2, 3) in https://arxiv.org/pdf/1607.01719.pdf
        Args:
            is_sqrt:
        """
        super().__init__()
        self.is_sqrt = is_sqrt

    def forward(self, source, target):
        d = source.size(1)
        ns, nt = source.size(0), target.size(0)

        # source covariance
        tmp_s = torch.ones((1, ns)).cuda() @ source
        cs = (source.t() @ source - (tmp_s.t() @ tmp_s) / ns) / (ns - 1)

        # target covariance
        tmp_t = torch.ones((1, nt)).cuda() @ target
        ct = (target.t() @ target - (tmp_t.t() @ tmp_t) / nt) / (nt - 1)

        # frobenius norm
        loss = (cs - ct).pow(2).sum()
        loss = loss.sqrt() if self.is_sqrt else loss
        loss = loss / (4 * d * d)

        return loss
