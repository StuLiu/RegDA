"""
@Project :
@File    : cutmix.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2023/12/25 15:07
@e-mail  : liuwa@hnu.edu.cn
"""
import numpy as np
import random
import cv2
import torch
import torch.nn.functional as tnf
from regda.utils.tools import index2onehot


def classmix(data_s, targets_s, data_t, targets_t, ratio=0.5, class_num=7, ignore_label=-1):
    """
    class mixing
    Args:
        data_s: [b, 3, h, w]
        targets_s: [b, 1, h, w]
        data_t: [b, 3, h, w]
        targets_t: [b, 1, h, w]
        ratio: 0~1
        class_num: number of classes

    Returns:
        mixed images and labels for both source and target domains
        data_s: [b, 3, h, w]
        targets_s: [b, h, w]
        data_t: [b, 3, h, w]
        targets_t: [b, h, w]
    """
    data_s, targets_s, data_t, targets_t = (data_s.clone(), targets_s.clone().long(),
                                            data_t.clone(), targets_t.clone().long())
    if targets_s.dim() == 3:
        targets_s = targets_s.unsqueeze(dim=1)
    if targets_t.dim() == 3:
        targets_t = targets_t.unsqueeze(dim=1)

    class_ids = torch.randperm(class_num)[: int(class_num * ratio)]  # rand batch-wise
    class_mix = torch.zeros((1, class_num, 1, 1), dtype=torch.int).cuda()   # (1, c, 1, 1)
    for c_id in class_ids:
        class_mix[:, c_id, :, :] = 1

    targets_s_onehot = index2onehot(targets_s, class_num=class_num, ignore_label=ignore_label)
    cond_mix = torch.sum(targets_s_onehot * class_mix, dim=1, keepdim=True).bool()
    targets_t[cond_mix] = targets_s[cond_mix]
    cond_mix = torch.broadcast_to(cond_mix, data_t.shape)
    data_t[cond_mix] = data_s[cond_mix]

    return data_s, targets_s.squeeze(dim=1), data_t, targets_t.squeeze(dim=1)


if __name__ == '__main__':
    image1 = cv2.imread('../../data/IsprsDA/Potsdam/img_dir/train/2_10_0_0_512_512.png')
    image1 = torch.Tensor(image1).cuda().int().permute(2, 0, 1).unsqueeze(dim=0).cuda()
    image1 = torch.cat([image1, image1], dim=0)

    image2 = cv2.imread('../../data/IsprsDA/Vaihingen/img_dir/train/area1_0_0_512_512.png')
    image2 = torch.Tensor(image2).cuda().int().permute(2, 0, 1).unsqueeze(dim=0).cuda()
    image2 = torch.cat([image2, image2], dim=0)

    label1 = cv2.imread('../../data/IsprsDA/Potsdam/ann_dir/train/2_10_0_0_512_512.png', cv2.IMREAD_UNCHANGED)
    label1 = torch.Tensor(label1).cuda().long().unsqueeze(dim=0)
    label1 = torch.cat([label1, label1], dim=0)

    label2 = cv2.imread('../../data/IsprsDA/Vaihingen/ann_dir/train/area1_0_0_512_512.png',
                        cv2.IMREAD_UNCHANGED)
    label2 = torch.Tensor(label2).cuda().long().unsqueeze(dim=0)
    label2 = torch.cat([label2, label2], dim=0)

    k = cv2.waitKey(1)

    while k != ord('q'):
        img1, lbl1, img2, lbl2 = classmix(image1, label1, image2, label2)
        img1 = img1.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        lbl1 = lbl1.cpu().numpy().astype(np.uint8)
        img2 = img2.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        lbl2 = lbl2.cpu().numpy().astype(np.uint8)
        cv2.imshow('i1', img1[0, :, :, :])
        cv2.imshow('i2', img1[1, :, :, :])
        cv2.imshow('i3', img2[0, :, :, :])
        cv2.imshow('i4', img2[1, :, :, :])
        cv2.imshow('l1', lbl1[0, :, :] * 20)
        cv2.imshow('l2', lbl1[1, :, :] * 20)
        cv2.imshow('l3', lbl2[0, :, :] * 20)
        cv2.imshow('l4', lbl2[1, :, :] * 20)

        k = cv2.waitKey(0)
