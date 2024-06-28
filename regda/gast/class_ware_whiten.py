"""
@Project : rsda
@File    : class_ware_whiten.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2023/3/31 下午1:37
@e-mail  : 1183862787@qq.com
"""
import torch
import torch.nn as nn
import torch.nn.functional as tnf


class ClassWareWhitening(nn.Module):

    def __init__(self, class_ids=(), groups=1):
        super().__init__()
        assert groups >= 1
        self.class_ids = class_ids
        self.groups = groups

    @staticmethod
    def get_covariance_matrix(feats, mask):
        """

        Args:
            feats: feature maps with shape=(b, k, h ,w)
            mask: mask for a pointed class. shape=(b, 1, h, w)
        Returns:
            covariance matrix for feats
        """
        num = torch.sum(mask)  # example number
        if num <= 1:
            return torch.eye(feats.shape[1]).cuda()
        x_masked = (feats * mask).permute(0, 2, 3, 1).reshape(-1, feats.shape[1])
        mask = mask.permute(0, 2, 3, 1).reshape(-1, 1)
        x_centered = x_masked - torch.sum(x_masked, dim=0, keepdim=True) * mask / num
        x_covariance = x_centered.t() @ x_centered / (num - 1)
        return x_covariance

    def instance_whitening_loss(self, feats, mask):
        x_cor = self.get_covariance_matrix(feats, mask)
        eye = torch.eye(feats.shape[1]).cuda()
        # print( ((x_cor - eye) ** 2).mean())
        return tnf.mse_loss(x_cor, eye)

    def forward(self, feats: torch.Tensor, labels: torch.Tensor):
        """
        Args:
            feats: deep features outputted by the encoder. [b, k, h, w]
            labels: labels, ground truth of source data or pseudo label of target data. [b, 1, h, w] or [b, h, w]
        Returns:
            sum of white loss for each classes
        """
        assert len(feats.shape) == 4 and len(labels.shape) >= 3
        assert feats.shape[1] % self.groups == 0
        if len(labels.shape) == 3:
            labels = labels.unsqueeze(dim=1)
        loss, step = 0, feats.shape[1] // self.groups
        for class_id in self.class_ids:
            mask_i = torch.where(labels == class_id, 1, 0)
            for group_id in range(self.groups):
                feats_group = feats[:, group_id * step: (group_id + 1) * step, :, :]
                loss += self.instance_whitening_loss(feats_group, mask_i)
        return loss


if __name__ == '__main__':
    a = [[2, 1, 3, 0],
         [5, 6, 7, 8],
         [1, 2, 3, 4],
         [2, 3, 4, 5],
         [0, 1, 0, 1],
         [5, 1, 3, 1]]
    _mask_i = [1, 0, 0, 1, 0, 0]
    fe = torch.FloatTensor(a).cuda().reshape(1, 1, 6, 4).permute(0, 3, 1, 2)
    mi = torch.LongTensor(_mask_i).cuda().reshape(1, 1, 6)
    cww = ClassWareWhitening(class_ids=[1, 2], groups=1)
    loss_cww = cww(fe, mi)   # 12.4375
    print(loss_cww)
