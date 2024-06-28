"""
@Project : rads2
@File    : DALoader.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2023/6/6 上午11:00
@e-mail  : 1183862787@qq.com
"""

import torch
from torch.utils.data import Dataset, DataLoader
import glob
import os
from skimage.io import imread
from albumentations.pytorch import ToTensorV2
from albumentations import HorizontalFlip, VerticalFlip, RandomRotate90, Normalize
from albumentations import OneOf, Compose
from ever.interface import ConfigurableMixin
from torch.utils.data import SequentialSampler, RandomSampler
from ever.api.data import CrossValSamplerGenerator
import numpy as np
import logging
from regda.datasets import LoveDA, IsprsDA


logger = logging.getLogger(__name__)


class DALoader(DataLoader, ConfigurableMixin):
    def __init__(self, config, datasets):
        ConfigurableMixin.__init__(self, config)
        dataset = eval(datasets)(image_dir=self.config.image_dir,
                                 mask_dir=self.config.mask_dir,
                                 transforms=self.config.transforms,
                                 label_type=self.config.label_type,
                                 read_sup=self.config.read_sup)

        if self.config.CV.i != -1:
            CV = CrossValSamplerGenerator(dataset, distributed=True, seed=2333)
            sampler_pairs = CV.k_fold(self.config.CV.k)
            train_sampler, val_sampler = sampler_pairs[self.config.CV.i]
            if self.config.training:
                sampler = train_sampler
            else:
                sampler = val_sampler
        else:
            sampler = RandomSampler(dataset) if self.config.training else SequentialSampler(
                dataset)

        super(DALoader, self).__init__(dataset,
                                       self.config.batch_size,
                                       sampler=sampler,
                                       num_workers=self.config.num_workers,
                                       pin_memory=self.config.pin_memory,
                                       drop_last=True
                                       )

    def set_default_config(self):
        self.config.update(dict(
            image_dir=None,
            mask_dir=None,
            batch_size=4,
            num_workers=4,
            pin_memory=True,
            scale_size=None,
            transforms=Compose([
                OneOf([
                    HorizontalFlip(True),
                    VerticalFlip(True),
                    RandomRotate90(True),
                ], p=0.75),
                Normalize(mean=(), std=(), max_pixel_value=1, always_apply=True),
                ToTensorV2()
            ]),
            label_type='id',
            read_sup=False,
        ))
