"""
@Project : Unsupervised_Domian_Adaptation
@File    : class_balance.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2023/3/5 下午3:59
@e-mail  : liuwa@hnu.edu.cn
"""
import torch
import torch.nn as nn
import torch.nn.functional as tnf
from tqdm import tqdm


def examples_cnt(dataloader, ignore_label=-1, save_prob=False):
    """

    Args:
        dataloader:
        ignore_label:
        save_prob:
    Returns:
        cnt: number of valid examples
        ratio: mean percent of valid examples in each image
    """
    cnt = 0.0
    cnt_all = 0.0
    for _, masks in tqdm(dataloader):
        lbl = masks['cls']
        if save_prob:
            lbl = torch.argmax(lbl, dim=1)
        cnt += torch.sum(lbl != ignore_label).detach().cpu().item()
        cnt_all += lbl.shape[0] * lbl.shape[1] * lbl.shape[2]
    ratio = cnt / cnt_all
    return cnt, ratio


def get_target_weight(cnt_s, ratio_s, cnt_t, ratio_t):
    weight = cnt_t * ratio_s / (cnt_s * ratio_t + 1e-7)
    return 1.0 if weight >= 1.0 else weight


class DomainBalance:

    def __init__(self, ignore_label=-1, decay=0.99, is_balance=True, cnt_s=1, cnt_t=1):
        super().__init__()
        self.ignore_label = ignore_label
        self.decay = decay
        self.is_balance = is_balance
        self.eps = 1e-7
        self.cnt_s = cnt_s
        self.cnt_t = cnt_t

    def get_target_weight(self):
        return self.valid_percent * self.cnt_t / (self.cnt_s + self.eps)

    @staticmethod
    def _ema(history, curr, decay):
        new_average = (1.0 - decay) * curr + decay * history
        return new_average
