"""
@Project : gstda
@File    : local_region_homog.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2023/12/18 下午4:58
@e-mail  : 1183862787@qq.com
"""
import os.path
import torch
import cv2
import sys
import warnings

import numpy as np
import skimage.io as iio
import torch.nn.functional as tnf
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch_scatter import scatter
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

warnings.filterwarnings('ignore')


class SAM:

    def __init__(self, sam_checkpoint="ckpts/sam_vit_h_4b8939.pth", model_type="vit_h"):
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.cuda()
        self.model = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,
            pred_iou_thresh=0.90,
            stability_score_thresh=0.95,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
        )

    def get_local_regions(self, image_path='data/IsprsDA/Vaihingen/img_dir/train/area1_0_0_512_512.png',
                          area_thrshold=1024, save=True, show=True) -> np.array:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        size = image.shape[0: 2]
        anns = self.model.generate(image)
        # print(len(anns))
        if show:
            self.show_anns(image, anns)

        mask = np.zeros(size)
        for i, ann in enumerate(anns):
            if ann['area'] >= area_thrshold:
                m = ann['segmentation']
                mask[m] = i + 1
        mask = mask.astype(np.int32)

        if save:
            save_path = image_path.replace('img_dir', 'reg_dir').replace('.png', '.tif')
            out_dir = os.path.split(save_path)[0]
            os.makedirs(out_dir, exist_ok=True)
            iio.imsave(save_path, mask)

        return mask

    @staticmethod
    def show_anns(image, anns) -> None:
        plt.figure(figsize=(20, 20))
        plt.imshow(image)

        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)

        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:, :, 3] = 0
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask
        ax.imshow(img)

        plt.axis('off')
        plt.show()


def get_all_regs(img_dir_tgt='datasets/potsdam/img_dir/train'):
    sam_model = SAM()
    img_files = os.listdir(img_dir_tgt)
    img_files.sort()
    for img_file in tqdm(img_files):
        if img_file.endswith('.png'):
            sam_model.get_local_regions(image_path=os.path.join(img_dir_tgt, img_file),
                                        save=True, show=False)


class Homogenizer(torch.nn.Module):

    def __init__(self, percent=0.9, class_num=6, ignore_label=255):
        super().__init__()
        self.percent = percent
        self.class_num = class_num
        self.ignore_label = ignore_label

    def _index2onehot(self, label):
        """Compute the one-hot label
        Args:
            label: torch.Tensor, gt or pseudo label, shape=(b, h, w)
        Returns:
            labels: (b*h*w, c)
        """
        labels = label.clone()
        if len(label.shape) == 4:
            labels = labels.squeeze(dim=1)
        assert labels.dim() == 3
        # b, h, w = labels.size()
        labels[labels == self.ignore_label] = self.class_num
        labels = labels.reshape(-1, )                           # (b*h*w, )
        labels = tnf.one_hot(labels, num_classes=self.class_num + 1)[:, :-1]  # (b*h*w, c)
        # labels = labels.reshape(b, h, w, c).permute(0, 3, 1, 2)
        return labels

    def forward(self, pseudo_labels, regions):
        """
        Args:
            pseudo_labels: Tensor, pseudo labels, (b, h, w)
            regions: Tensor, superpixel label, (b, h, w)
        Returns:
            pseudo_labels: Tensor, refined pseudo labels, (b, h, w)
        """
        assert pseudo_labels.dim() == 3
        b, h, w = pseudo_labels.size()
        label_t_onehot = self._index2onehot(pseudo_labels)              # (b*h*w, c)
        label_t_onehot = label_t_onehot.reshape(b, -1, self.class_num)  # (b, h*w, c)
        regions_ = regions.reshape(b, -1, 1)                            # (b, h*w, 1)

        # region statistic
        class_num_4sup = scatter(src=label_t_onehot, index=regions_, dim=1, reduce='sum')   # (b, n_sup, c)
        pixel_num_4sup = torch.sum(class_num_4sup, dim=-1, keepdim=True)                    # (b, n_sup, 1)
        class_num_max, class_num_max_id = torch.max(class_num_4sup, dim=-1, keepdim=True)   # (b, n_sup, 1)
        percent_maxclass = class_num_max / (pixel_num_4sup + 1e-5)                          # (b, n_sup, 1)
        class_num_max_id[percent_maxclass < self.percent] = self.ignore_label               # (b, n_sup, 1)

        # local region homogenizing
        pseudo_labels_ = torch.gather(class_num_max_id, dim=1, index=regions_)  # (b, h*w, 1)
        pseudo_labels_ = pseudo_labels_.reshape(b, h, w)                # (b, h, w)
        pseudo_labels_[regions == 0] = self.ignore_label                        # (b, h, w)
        # pseudo_labels_lrh = pseudo_labels_
        pseudo_labels_lrh = torch.where(pseudo_labels_.eq(self.ignore_label), pseudo_labels, pseudo_labels_)
        return pseudo_labels_lrh


if __name__ == '__main__':
    # # sam_model = SAM(sam_checkpoint="ckpts/sam_vit_h_4b8939.pth", model_type="vit_h")
    # # sam_model.get_local_regions(save=False, show=True)
    pseudo_labels = torch.randint(0, 3, (2, 3, 3))
    pseudo_labels[0,0,0] = 255
    regions = torch.zeros((2, 3, 3)).long()
    regions[:, 1: 2, :] = 1
    regions[:,2:3,:] = 2
    her = Homogenizer(3, 255)
    out = her(pseudo_labels, regions)
    print(pseudo_labels)
    print(regions)
    print(out)
    # get_all_regs(img_dir_tgt='datasets/vaihingen/img_dir/train')
    # get_all_regs(img_dir_tgt='datasets/potsdam/img_dir/train')
