"""
@Project : rsda, SAN-SAW
@File    : SAW.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2023/3/29 下午5:15
@e-mail  : 1183862787@qq.com
"""
# https://openaccess.thecvf.com/content/CVPR2022/html/Peng_Semantic-Aware_Domain_Generalized_Segmentation_CVPR_2022_paper.html
import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class SAW(nn.Module):

    def __init__(self, classifyer, selected_classes, relax_denom=2.0):
        """Semantic aware whitening. Reference from https://github.com/leolyj/SAN-SAW, and its paper
            https://openaccess.thecvf.com/content/CVPR2022/html/Peng_Semantic-Aware_Domain_Generalized_Segmentation_CVPR_2022_paper.html
        Args:
            selected_classes:
            relax_denom:
        """
        super(SAW, self).__init__()
        self.classifyer = classifyer
        self.selected_classes = selected_classes
        self.C = len(selected_classes)
        assert self.C in [2, 4, 6, 8, 16]
        self.i = torch.eye(self.C, self.C).cuda()
        self.reversal_i = torch.ones(self.C, self.C).float().triu(diagonal=1).cuda()
        self.num_off_diagonal = torch.sum(self.reversal_i)
        if relax_denom == 0:
            print("Note relax_denom == 0!")
            self.margin = 0
        else:
            self.margin = self.num_off_diagonal // relax_denom

    def get_mask_matrix(self):
        return self.i, self.reversal_i, self.margin, self.num_off_diagonal

    @staticmethod
    def get_covariance_matrix(x, eye=None):
        eps = 1e-5
        B, C, H, W = x.shape  # i-th feature size (B X C X H X W)
        HW = H * W
        if eye is None:
            eye = torch.eye(C).cuda()
        x = x.contiguous().view(B, C, -1)  # B X C X H X W > B X C X (H X W)
        x_cor = torch.bmm(x, x.transpose(1, 2)).div(HW - 1) + (eps * eye)  # C X C / HW

        return x_cor, B

    def instance_whitening_loss(self, x, eye, mask_matrix, margin, num_remove_cov):
        x_cor, B = self.get_covariance_matrix(x, eye=eye)
        x_cor_masked = x_cor * mask_matrix

        off_diag_sum = torch.sum(torch.abs(x_cor_masked), dim=(1, 2), keepdim=True) - margin  # B X 1 X 1
        loss = torch.clamp(torch.div(off_diag_sum, num_remove_cov), min=0)  # B X 1 X 1
        loss = torch.sum(loss) / B

        return loss

    def sort_with_idx(self, x, idx, weights):
        b, c, _, _ = x.size()
        after_sort = torch.zeros_like(x)
        weights = F.sigmoid(weights)
        for i in range(b):

            for k in range(int(c / self.C)):
                for j in range(self.C):
                    channel_id = idx[self.selected_classes[j]][k]
                    wgh = weights[self.selected_classes[j]][channel_id]
                    after_sort[i][self.C * k + j][:][:] = wgh * x[i][channel_id][:][:]

        return after_sort

    def forward(self, x):
        weights_t = None
        weights_keys = self.classifyer.state_dict().keys()
        selected_keys_classify = []

        for key in weights_keys:
            if "weight" in key:
                selected_keys_classify.append(key)

        for key in selected_keys_classify:
            weights_t = self.classifyer.state_dict()[key]

        classsifier_weights = abs(weights_t.squeeze())
        _, index = torch.sort(classsifier_weights, descending=True, dim=1)
        f_map_lst = []
        B, channel_num, H, W = x.shape
        x = self.sort_with_idx(x, index, classsifier_weights)

        for i in range(int(channel_num / self.C)):
            group = x[:, self.C * i:self.C * (i + 1), :, :]
            f_map_lst.append(group)

        eye, mask_matrix, margin, num_remove_cov = self.get_mask_matrix()
        SAW_loss = torch.FloatTensor([0]).cuda()

        for i in range(int(channel_num / self.C)):
            loss = self.instance_whitening_loss(f_map_lst[i], eye, mask_matrix, margin, num_remove_cov)
            SAW_loss = SAW_loss + loss

        return SAW_loss
