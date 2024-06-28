"""
@Project : Unsupervised_Domian_Adaptation
@File    : augmentation.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2023/4/25 下午9:59
@e-mail  : 1183862787@qq.com
"""

import math
import numpy as np
import random
from PIL import Image

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target=None, mask_sup=None):
        image = F.resize(image, self.size)
        if target is not None:
            target = F.resize(target, self.size, interpolation=F.InterpolationMode.NEAREST)
        if mask_sup is not None:
            mask_sup = F.resize(mask_sup, self.size, interpolation=F.InterpolationMode.NEAREST)
        return image, target, mask_sup


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target=None, mask_sup=None):
        if random.random() < self.prob:
            image = F.hflip(image)
            if target is not None:
                target = F.hflip(target)
            if mask_sup is not None:
                mask_sup = F.hflip(mask_sup)
        return image, target, mask_sup


class RandomVerticalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target=None, mask_sup=None):
        if random.random() < self.prob:
            image = F.vflip(image)
            if target is not None:
                target = F.vflip(target)
            if mask_sup is not None:
                mask_sup = F.vflip(mask_sup)
        return image, target, mask_sup


class RandomRotate90(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target=None, mask_sup=None):
        if random.random() < self.prob:
            image = torch.rot90(image, k=1, dims=[1, 2])    # rotate neg 90
            if target is not None:
                target = torch.rot90(target, k=1, dims=[1, 2])
            if mask_sup is not None:
                mask_sup = torch.rot90(mask_sup, k=1, dims=[1, 2])
        return image, target, mask_sup


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target, mask_sup=None):
        crop_params = T.RandomCrop.get_params(image, self.size)
        image = F.crop(image, *crop_params)
        if target is not None:
            target = F.crop(target, *crop_params)
        if mask_sup is not None:
            mask_sup = F.crop(mask_sup, *crop_params)
        return image, target, mask_sup


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target, mask_sup=None):
        image = F.center_crop(image, self.size)
        if target is not None:
            target = F.center_crop(target, self.size)
        if mask_sup is not None:
            mask_sup = F.center_crop(mask_sup, self.size)
        return image, target, mask_sup


class Normalize2(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target, mask_sup=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target, mask_sup


class Normalize(object):
    def __init__(self, mean, std, clamp=False):
        self.mean = mean
        self.std = std
        self.clamp = clamp

    def __call__(self, image, target, mask_sup=None):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if self.clamp:
            image = torch.clamp(image, max=1.0)
        return image, target, mask_sup


class Pad(object):
    def __init__(self, padding_n, padding_fill_value=0, padding_fill_target_value=0):
        self.padding_n = padding_n
        self.padding_fill_value = padding_fill_value
        self.padding_fill_target_value = padding_fill_target_value

    def __call__(self, image, target, mask_sup=None):
        image = F.pad(image, self.padding_n, self.padding_fill_value)
        if target is not None:
            target = F.pad(target, self.padding_n, self.padding_fill_target_value)
        if mask_sup is not None:
            mask_sup = F.pad(mask_sup, self.padding_n, self.padding_fill_target_value)
        return image, target, mask_sup


class ToTensor(object):
    def __call__(self, image, target, mask_sup=None):
        image = torch.as_tensor(image, dtype=torch.float32)
        if target is not None:
            target = torch.as_tensor(np.array(target), dtype=torch.int64)
        if mask_sup is not None:
            mask_sup = torch.as_tensor(np.array(mask_sup), dtype=torch.int64)
        return image, target, mask_sup


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask=None, mask_sup=None):
        for t in self.transforms:
            image, mask, mask_sup = t(image, mask, mask_sup)
        return {'image': image, 'mask': mask, 'mask_sup': mask_sup}


# 使用：
if __name__ == '__main__':
    import ever as er
    transform = Compose([
        RandomCrop((512, 512)),
        RandomHorizontalFlip(prob=1),
        RandomVerticalFlip(prob=1),
        RandomRotate90(prob=1),
        Normalize(
            mean=(73.53223948, 80.01710095, 74.59297778),
            std=(41.5113661, 35.66528876, 33.75830885)
        ),
    ])

    _input = torch.ones([3, 1024, 1024]) * 128
    a = list(range(0, 1024))
    _label = torch.FloatTensor(a).unsqueeze(dim=0).unsqueeze(dim=0).expand(7, 1024, 1024)
    # _label = torch.ones([8, 7, 1024, 1024])
    _out = transform(image=_input, mask=_label)
    print(_out['image'], _out['mask'])
    print(_out['image'].shape, _out['mask'].shape)
