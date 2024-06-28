"""
@Project : Unsupervised_Domian_Adaptation
@File    : superpixels.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2023/10/2 上午12:00
@e-mail  : 1183862787@qq.com
"""
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import skimage.io as iio
import warnings

from tqdm import tqdm
from glob import glob
# from module.gast.sin.spixel_deconv import SpixelNet1l_bn


warnings.filterwarnings("ignore")

# def fortestLSC():
#     img = cv2.imread("../../data/IsprsDA/Potsdam/img_dir/train/2_10_0_0_512_512.png")
#     # cv2.imshow('origin', img)
#     # cv2.waitKey(0)
#
#     # 初始化supixl项，超像素平均尺寸20（默认为10），平滑因子20
#     supixl = cv2.ximgproc.createSuperpixelLSC(img, region_size=16, ratio=0.075)                ### LSC 算法
#     # supixl = cv2.ximgproc.createSuperpixelSLIC(img, region_size=10, ruler=30)   ### SLIC 算法
#
#     supixl.iterate(30)                                # 迭代次数，越大效果越好
#     mask_supixl = supixl.getLabelContourMask()          # 获取Mask，超像素边缘Mask==1
#     label_supixl = supixl.getLabels()                   # 获取超像素标签
#     number_supixl = supixl.getNumberOfSuperpixels()     # 获取超像素数目
#     mask_inv_supixl = cv2.bitwise_not(mask_supixl)
#     img_supixl = cv2.bitwise_and(img, img, mask=mask_inv_supixl) #在原图上绘制超像素边界
#
#     color_img = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
#     color_img[:] = (0, 255 , 0)
#     result_ = cv2.bitwise_and(color_img, color_img, mask=mask_supixl)
#     result = cv2.add(img_supixl, result_)
#     cv2.imshow('sp', result)
#     cv2.waitKey(0)
#     cv2.imwrite("./cat_supixl.png", result)


class SuperPixelsLSC(object):

    def __init__(self, region_size=16, ratio=0.075, iterate_num=100):
        self.region_size = region_size
        self.ratio = ratio
        self.iterate_num = iterate_num

    def get_super_pixels(self, img_ndarray):
        """

        Args:
            img_ndarray: np.ndarray, image read by cv2

        Returns:
            label_tensor, torch.Tensor, the label ids of superpixels, shape=(h, w)
            number_supixl: int, the number of superpixels
        """
        # LSC 算法
        supixl = cv2.ximgproc.createSuperpixelLSC(img_ndarray, region_size=self.region_size, ratio=self.ratio)
        supixl.iterate(self.iterate_num)  # 迭代次数，越大效果越好
        label_supixl = supixl.getLabels()  # 获取超像素标签
        number_supixl = supixl.getNumberOfSuperpixels()  # 获取超像素数目

        # 在原图上绘制超像素边界
        edge_supixl = supixl.getLabelContourMask()  # 获取Mask，超像素边缘Mask==255
        mask_inv_supixl = cv2.bitwise_not(edge_supixl)
        _img = img_ndarray.copy()
        img_supixl = cv2.bitwise_and(_img, _img, mask=mask_inv_supixl)

        color_img = np.zeros((_img.shape[0], _img.shape[1], 3), np.uint8)
        color_img[:] = (0, 255, 0)
        color_img = cv2.bitwise_and(color_img, color_img, mask=edge_supixl)
        color_img = cv2.add(img_supixl, color_img)

        return number_supixl, label_supixl, color_img


def get_superpixels(dir_path, out_dir, postfix='png', show=False, shrinking=True):
    """
    Get superpixel labels and visualizations
    Args:
        shrinking:
        dir_path: dir path to target training images
        out_dir
        postfix: 'png'
        show: Boolean, vis window if True
    Returns:
        None
    """
    img_paths = glob(r'' + dir_path + '/*.' + postfix)
    img_paths.sort()
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir + "_vis", exist_ok=True)
    if shrinking:
        os.makedirs(out_dir + "_shrink", exist_ok=True)
        os.makedirs(out_dir + "_shrink_vis", exist_ok=True)

    sp = SuperPixelsLSC(region_size=16, ratio=0.075, iterate_num=100)

    for img_path in tqdm(img_paths):
        img_bgr = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)[:, :, :3]
        number_supixl, label_supixl, color_img = sp.get_super_pixels(img_bgr)
        # cv2.imwrite(os.path.join(out_dir, str(os.path.basename(img_path)).replace(f'.{postfix}', '.tif')),
        #             label_supixl.astype(np.uint16))
        iio.imsave(os.path.join(out_dir, str(os.path.basename(img_path)).replace(f'.{postfix}', '.tif')),
                   label_supixl)
        # label_supixl_ = iio.imread(
        #     os.path.join(out_dir, str(os.path.basename(img_path)).replace(f'.{postfix}', '.tif'))
        # )

        if shrinking:
            edge_shrinking(out_dir, os.path.basename(img_path), postfix, label_supixl, win_size=3)

        cv2.imwrite(os.path.join(out_dir + "_vis", os.path.basename(img_path)), color_img)

        if show:
            cv2.imshow('sup_image', color_img)
            cv2.waitKey(0)


def edge_shrinking(out_dir, img_name, postfix, label_supixl, win_size=3, region_size=16):
    h, w = label_supixl.shape
    cnt_sup = int(h / region_size * w / region_size)
    keeped_mat = np.ones_like(label_supixl)
    for i in range(h):
        for j in range(w):
            curr_val = label_supixl[i][j]
            keeped = True
            for shift_i in range(-win_size, win_size + 1):
                for shift_j in range(-win_size, win_size + 1):
                    new_i, new_j = i + shift_i, j + shift_j
                    if new_i < 0 or new_i >= h or new_j < 0 or new_j >= w:
                        continue
                    temp_val = label_supixl[new_i][new_j]
                    if temp_val != curr_val:
                        keeped = False
                        keeped_mat[i][j] = 0
                        break
                if not keeped:
                    break
    label_supixl_out = np.where(keeped_mat == 1, label_supixl, cnt_sup)
    iio.imsave(os.path.join(out_dir + '_shrink', img_name.replace(f'.{postfix}', '.tif')),
               label_supixl_out)
    return label_supixl_out


if __name__ == '__main__':
    # sp = SuperPixelsLSC()
    # img_cv2_1 = cv2.imread("../../data/IsprsDA/Potsdam/img_dir/train/2_10_0_0_512_512.png")
    # img_cv2_2 = cv2.imread("../../data/IsprsDA/Potsdam/img_dir/train/2_10_0_512_512_1024.png")

    get_superpixels(dir_path="../../data/IsprsDA/Vaihingen/img_dir/train",
                    out_dir="../../data/IsprsDA/Vaihingen/ann_dir/train_sup",
                    postfix='png', show=False)
    get_superpixels(dir_path="../../data/IsprsDA/Potsdam/img_dir/train",
                    out_dir="../../data/IsprsDA/Potsdam/ann_dir/train_sup",
                    postfix='png', show=False)
    get_superpixels(dir_path="../../data/LoveDA/Val/Rural/images_png",
                    out_dir="../../data/LoveDA/Val/Rural/masks_png_sup",
                    postfix='png', show=False)
    get_superpixels(dir_path="../../data/LoveDA/Val/Urban/images_png",
                    out_dir="../../data/LoveDA/Val/Urban/masks_png_sup",
                    postfix='png', show=False)
