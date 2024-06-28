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


def cutmix(data_s, targets_s, data_t, targets_t, alpha=1.0):
    data_s, targets_s, data_t, targets_t = data_s.clone(), targets_s.clone(), data_t.clone(), targets_t.clone()
    lam = np.random.beta(alpha, alpha)

    image_h, image_w = data_s.shape[2:]
    cx = np.random.uniform(0, image_w)
    cy = np.random.uniform(0, image_h)
    w = image_w * np.sqrt(1 - lam)
    h = image_h * np.sqrt(1 - lam)
    x0 = int(np.round(max(cx - w / 2, 0)))
    x1 = int(np.round(min(cx + w / 2, image_w)))
    y0 = int(np.round(max(cy - h / 2, 0)))
    y1 = int(np.round(min(cy + h / 2, image_h)))

    data_t[:, :, y0:y1, x0:x1] = data_s[:, :, y0:y1, x0:x1]
    targets_t[:, y0:y1, x0:x1] = targets_s[:, y0:y1, x0:x1]
    return data_s, targets_s, data_t, targets_t

def cutmix2(data, targets, alpha=1.0):
    data, targets = data.clone(), targets.clone()
    indices = torch.randperm(data.size(0))  # rand batch-wise
    shuffled_data = data[indices]
    shuffled_targets = targets[indices]

    lam = np.random.beta(alpha, alpha)

    image_h, image_w = data.shape[2:]
    cx = np.random.uniform(0, image_w)
    cy = np.random.uniform(0, image_h)
    w = image_w * np.sqrt(1 - lam)
    h = image_h * np.sqrt(1 - lam)
    x0 = int(np.round(max(cx - w / 2, 0)))
    x1 = int(np.round(min(cx + w / 2, image_w)))
    y0 = int(np.round(max(cy - h / 2, 0)))
    y1 = int(np.round(min(cy + h / 2, image_h)))

    data[:, :, y0:y1, x0:x1] = shuffled_data[:, :, y0:y1, x0:x1]
    targets[:, y0:y1, x0:x1] = shuffled_targets[:, y0:y1, x0:x1]
    return data, targets


if __name__ == '__main__':
    image1 = cv2.imread('1.png')
    image1 = torch.Tensor(image1).int().permute(2, 0, 1).unsqueeze(dim=0)
    image1 = torch.cat([image1, image1], dim=0)

    image2 = cv2.imread('2.png')
    image2 = torch.Tensor(image2).int().permute(2, 0, 1).unsqueeze(dim=0)
    image2 = torch.cat([image2, image2], dim=0)

    label1 = torch.zeros_like(image1[:, 0, :, :])
    label2 = torch.ones_like(image2[:, 0, :, :])
    k = cv2.waitKey(1)

    while k != ord('q'):
        img1, lbl1, img2, lbl2 = cutmix(image1, label1, image2, label2, alpha=1)
        img1 = img1.permute(0, 2, 3, 1).numpy().astype(np.uint8)
        lbl1 = lbl1.numpy().astype(np.uint8)
        img2 = img2.permute(0, 2, 3, 1).numpy().astype(np.uint8)
        lbl2 = lbl2.numpy().astype(np.uint8)
        cv2.imshow('i1', img1[0, :, :, :])
        cv2.imshow('i2', img1[1, :, :, :])
        cv2.imshow('i3', img2[0, :, :, :])
        cv2.imshow('i4', img2[1, :, :, :])
        cv2.imshow('l1', lbl1[0, :, :] * 255)
        cv2.imshow('l2', lbl1[1, :, :] * 255)
        cv2.imshow('l3', lbl2[0, :, :] * 255)
        cv2.imshow('l4', lbl2[1, :, :] * 255)

        k = cv2.waitKey(0)
