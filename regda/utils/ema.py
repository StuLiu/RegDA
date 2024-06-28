"""
@Project : Unsupervised_Domian_Adaptation
@File    : ema.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2023/7/22 上午11:48
@e-mail  : 1183862787@qq.com
"""
import torch
import torch.nn as nn


# class ExponentialMovingAverage():
#     def __init__(self, decay):
#         self.decay = decay
#         self.shadow_params = {}
#
#     def register(self, model):
#         for name, param in model.named_parameters():
#             if param.requires_grad:
#                 self.shadow_params[name] = param.data.clone()
#
#     def update(self, model):
#         for name, param in model.named_parameters():
#             if param.requires_grad:
#                 self.shadow_params[name] -= (1 - self.decay) * (self.shadow_params[name] - param.data)
#
#     def apply(self, model):
#         for name, param in model.named_parameters():
#             if param.requires_grad:
#                 param.data = self.shadow_params[name]


class ExponentialMovingAverage:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
