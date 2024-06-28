"""
@Project :
@File    :
@IDE     : PyCharm
@Author  : Wang Liu
@Date    :
@e-mail  : 1183862787@qq.com
"""

import torch
from torch.utils.data import Dataset#, DataLoader
import glob
import os
from skimage.io import imread
# from albumentations.pytorch import ToTensorV2
# from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90, Normalize
# from albumentations import OneOf, Compose
# from ever.interface import ConfigurableMixin
# from torch.utils.data import SequentialSampler, RandomSampler
# from ever.api.data import CrossValSamplerGenerator
import numpy as np
import logging


logger = logging.getLogger(__name__)


class BaseData(Dataset):
    def __init__(self, image_dir, mask_dir, transforms=None,
                 label_type='id', offset=-1, ignore_label=-1, num_class=7, read_sup=False):
        assert label_type in ['id', 'prob']
        self.label_type = label_type
        self.n_classes = num_class
        self.ignore_label = ignore_label
        self.offset = offset
        self.read_sup = read_sup
        self.rgb_filepath_list = []
        self.cls_filepath_list = []
        self.sup_filepath_list = []
        if isinstance(image_dir, list):
            for img_dir_path, mask_dir_path in zip(image_dir, mask_dir):
                self.batch_generate(img_dir_path, mask_dir_path)

        else:
            self.batch_generate(image_dir, mask_dir)

        self.transforms = transforms

    def batch_generate(self, image_dir, mask_dir):
        rgb_filepath_list = glob.glob(os.path.join(image_dir, '*.tif'))
        rgb_filepath_list += glob.glob(os.path.join(image_dir, '*.png'))

        logger.info('Dataset images: %d' % len(rgb_filepath_list))
        
        rgb_filename_list = [os.path.split(fp)[-1] for fp in rgb_filepath_list]
        cls_filepath_list = []
        sup_filepath_list = []
        for fname in rgb_filename_list:
            if mask_dir is not None:
                cls_filepath_list.append(os.path.join(mask_dir, fname))
            # a = os.path.join(mask_dir + '_sup', f"{fname.split('.')[0]}.tif")
            sup_filepath_list.append(os.path.join(image_dir.replace('img_dir', 'reg_dir'),
                                                  f"{fname.split('.')[0]}.tif"))
        self.rgb_filepath_list += rgb_filepath_list
        self.cls_filepath_list += cls_filepath_list
        self.sup_filepath_list += sup_filepath_list

    def __getitem__(self, idx):
        # read image
        image = imread(self.rgb_filepath_list[idx])
        if self.label_type == 'prob':
            image = torch.from_numpy(image).float().permute(2, 0, 1)

        # read superpixel label
        mask_sup = None
        if self.read_sup:
            mask_sup = imread(self.sup_filepath_list[idx]).astype(np.int64)
            mask_sup = torch.from_numpy(mask_sup).unsqueeze(dim=0).long()

        if len(self.cls_filepath_list) > 0:
            if self.label_type == 'id':
                # 0~7 --> -1~6, 0 in mask.png represents the black area in the input.png
                mask = imread(self.cls_filepath_list[idx]).astype(np.int64) + self.offset
            else:
                # mask = torch.from_numpy(np.load(f'{self.cls_filepath_list[idx]}.npy')).float()
                mask = torch.load(f'{self.cls_filepath_list[idx]}.pt', map_location=torch.device('cpu'))
            # avoid noise label
            mask[mask >= self.n_classes] = self.ignore_label

            # data augmentation
            if self.transforms is not None:
                if self.read_sup:
                    blob = self.transforms(image=image, mask=mask, mask_sup=mask_sup)
                else:
                    blob = self.transforms(image=image, mask=mask)
                image = blob['image']
                mask = blob['mask']
                if self.read_sup:
                    mask_sup = blob['mask_sup']

            if self.read_sup:
                return image, dict(cls=mask, sup=mask_sup, fname=os.path.basename(self.rgb_filepath_list[idx]))
            else:
                return image, dict(cls=mask, fname=os.path.basename(self.rgb_filepath_list[idx]))
        else:
            if self.transforms is not None:
                blob = self.transforms(image=image, mask=None, mask_sup=mask_sup if self.read_sup else None)
                image = blob['image']
                if self.read_sup:
                    mask_sup = blob['mask_sup']
            if self.read_sup:
                return image, dict(sup=mask_sup, fname=os.path.basename(self.rgb_filepath_list[idx]))
            else:
                return image, dict(fname=os.path.basename(self.rgb_filepath_list[idx]))


    def __len__(self):
        return len(self.rgb_filepath_list)
