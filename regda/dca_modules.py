# -*- coding:utf-8 -*-

# @Filename: my_modules
# @Project : Unsupervised_Domian_Adaptation
# @date    : 2021-12-06 15:34
# @Author  : Linshan

import torch.nn as nn
import torch.nn.functional as F
import torch
from audtorch.metrics.functional import pearsonr


class CategoryAlign_Module(nn.Module):
    def __init__(self, num_classes=7, ignore_bg=False):
        super(CategoryAlign_Module, self).__init__()
        self.num_classes = num_classes
        self.ignore_bg = ignore_bg

    def get_context(self, preds, feats):
        b, c, h, w = feats.size()
        _, num_cls, _, _ = preds.size()

        # softmax preds
        assert preds.max() <= 1 and preds.min() >= 0, print(preds.max(), preds.min())
        preds = preds.view(b, num_cls, 1, h * w)  # (b, num_cls, 1, hw)
        feats = feats.view(b, 1, c, h * w)  # (b, 1, c, hw)

        vectors = (feats * preds).sum(-1) / preds.sum(-1)  # (b, num_cls, C)

        if self.ignore_bg:
            vectors = vectors[:, 1:, :]  # ignore the background
        vectors = F.normalize(vectors, dim=1)
        return vectors

    def get_intra_corcoef_mat(self, preds, feats):
        context = self.get_context(preds, feats).mean(0)

        n, c = context.size()
        mat = torch.zeros([n, n]).to(context.device)
        for i in range(n):
            for j in range(n):
                cor = pearsonr(context[i, :], context[j, :])
                mat[i, j] += cor[0]
        return mat

    def get_cross_corcoef_mat(self, preds1, feats1, preds2, feats2):
        context1 = self.get_context(preds1, feats1).mean(0)
        context2 = self.get_context(preds2, feats2).mean(0)

        n, c = context1.size()
        mat = torch.zeros([n, n]).to(context1.device)
        for i in range(n):
            for j in range(n):
                cor = pearsonr(context1[i, :], context2[j, :])
                mat[i, j] += cor[0]
        return mat

    def regularize(self, cor_mat):
        n = self.num_classes - 1 if self.ignore_bg else self.num_classes
        assert cor_mat.size()[0] == n

        # label = (torch.ones([n, n]) * -1).to(cor_mat.device)
        # diag = torch.diag_embed(torch.Tensor([2]).repeat(1, n)).to(cor_mat.device)
        # label = (label + diag).view(n, n)
        #
        # # loss = - torch.log(torch.clamp(label * cor_mat, min=1e-6))
        # loss = (1 - label*cor_mat).pow(2)
        pos = - torch.log(torch.diag(cor_mat)).mean()
        undiag = cor_mat.flatten()[:-1].view(n-1, n+1)[:, 1:].flatten()
        # undiag_clone = undiag.clone()
        low = torch.Tensor([1e-6]).to(undiag.device)
        neg = - torch.log(1 - undiag.max(low)).mean()

        loss = pos + neg
        return loss


def ICR(inputs, num_classes, multi_layer=False, ignore_bg=True):
    """
        Intra-domain Covariance Regularization
    """
    m = CategoryAlign_Module(ignore_bg=ignore_bg, num_classes=num_classes)

    if multi_layer:
        preds1, preds2, feats = inputs

        B = preds1.size()[0]
        preds = ((preds1.softmax(dim=1) + preds2.softmax(dim=1))/2).detach()

        preds1, feats1 = preds[:B // 2, :, :, :], feats[:B // 2, :, :, :]
        preds2, feats2 = preds[B // 2:, :, :, :], feats[B // 2:, :, :, :]
        cor_mat = m.get_cross_corcoef_mat(preds1, feats1, preds2, feats2)
        loss = m.regularize(cor_mat)
    else:
        preds, feats = inputs
        # cor_mat = m.get_intra_corcoef_mat(preds, feats)

        B = preds.size()[0]
        preds = preds.softmax(dim=1).detach()
        preds1, feats1 = preds[:B // 2, :, :, :], feats[:B // 2, :, :, :]
        preds2, feats2 = preds[B // 2:, :, :, :], feats[B // 2:, :, :, :]
        cor_mat = m.get_cross_corcoef_mat(preds1, feats1, preds2, feats2)
        loss = m.regularize(cor_mat)
    return loss


def CCR(source, target, num_classes, multi_layer=False, ignore_bg=True):
    """
        Cross-domain Covariance Regularization
    """
    m = CategoryAlign_Module(ignore_bg=ignore_bg, num_classes=num_classes)

    if multi_layer:
        S_preds1, S_preds2, S_feats = source
        T_preds1, T_preds2, T_feats = target

        S_preds = ((S_preds1.softmax(dim=1) + S_preds2.softmax(dim=1))/2)
        T_preds = ((T_preds1.softmax(dim=1) + T_preds2.softmax(dim=1))/2)

        cor_mat = m.get_cross_corcoef_mat(S_preds.detach(), S_feats.detach(),
                                          T_preds.detach(), T_feats)
        loss = m.regularize(cor_mat)

    else:
        S_preds, S_feats = source
        T_preds, T_feats = target
        S_preds, T_preds = S_preds.softmax(dim=1), T_preds.softmax(dim=1)

        cor_mat = m.get_cross_corcoef_mat(S_preds.detach(), S_feats.detach(),
                                          T_preds.detach(), T_feats)
        loss = m.regularize(cor_mat)
    return loss


def MSE_intra(inputs, multi_layer=False, ignore_bg=True):
    """
        MSE-Alignment for intra domain
    """
    m = CategoryAlign_Module(ignore_bg=ignore_bg)

    if multi_layer:
        preds1, preds2, feats = inputs

        B = preds1.size()[0]
        preds = ((preds1.softmax(dim=1) + preds2.softmax(dim=1))/2).detach()
        preds1, feats1 = preds[:B // 2, :, :, :], feats[:B // 2, :, :, :]
        preds2, feats2 = preds[B // 2:, :, :, :], feats[B // 2:, :, :, :]
        context1, context2 = m.get_context(preds1, feats1), m.get_context(preds2, feats2)
        loss = F.mse_loss(context1, context2, reduction='mean')
    else:
        preds, feats = inputs
        B = preds.size()[0]
        preds = preds.softmax(dim=1).detach()
        preds1, feats1 = preds[:B // 2, :, :, :], feats[:B // 2, :, :, :]
        preds2, feats2 = preds[B // 2:, :, :, :], feats[B // 2:, :, :, :]
        context1, context2 = m.get_context(preds1, feats1), m.get_context(preds2, feats2)
        loss = F.mse_loss(context1, context2, reduction='mean')
    return loss


def MSE_cross(source, target, multi_layer=False, ignore_bg=True):
    """
        MSE-Alignment for cross domain
    """
    m = CategoryAlign_Module(ignore_bg=ignore_bg)

    if multi_layer:
        S_preds1, S_preds2, S_feats = source
        T_preds1, T_preds2, T_feats = target

        S_preds = ((S_preds1.softmax(dim=1) + S_preds2.softmax(dim=1))/2)
        T_preds = ((T_preds1.softmax(dim=1) + T_preds2.softmax(dim=1))/2)

        context1, context2 = m.get_context(S_preds.detach(), S_feats.detach()), \
                             m.get_context(T_preds.detach(), T_feats)
        loss = F.mse_loss(context1, context2, reduction='mean')

    else:
        S_preds, S_feats = source
        T_preds, T_feats = target

        S_preds, T_preds = S_preds.softmax(dim=1), T_preds.softmax(dim=1)

        context1, context2 = m.get_context(S_preds.detach(), S_feats.detach()), \
                             m.get_context(T_preds.detach(), T_feats)
        loss = F.mse_loss(context1, context2, reduction='mean')
    return loss


if __name__ == '__main__':
    num_classes = 7
    mat = torch.ones([num_classes, num_classes]) * -1
    print(mat.shape)
    diag = torch.diag_embed(torch.Tensor([2]).repeat(1, num_classes))
    print(diag)
    print(mat + diag)

