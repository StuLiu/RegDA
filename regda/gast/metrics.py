"""
@Project : rads2
@File    : metrics.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2023/8/1 上午10:44
@e-mail  : 1183862787@qq.com
"""
import logging
import os
import time
import numpy as np
import prettytable as pt
import pandas as pd
from ever.api.metric.pixel import PixelMetric
import math


class PixelMetricIgnore(PixelMetric):

    def __init__(self, num_classes, logdir=None, logger=None, class_names=None, ignore_labels=list()):
        super().__init__(num_classes, logdir, logger, class_names)
        self.ignore_labels = ignore_labels
        self.ignore_labels.sort(reverse=True)

    def summary_all(self, dec=5):
        dense_cm = self._total.toarray()

        iou_per_class = np.round(PixelMetric.compute_iou_per_class(dense_cm), dec).tolist()
        F1_per_class = np.round(PixelMetric.compute_F_measure_per_class(dense_cm, beta=1.0), dec).tolist()
        precision_per_class = np.round(PixelMetric.compute_precision_per_class(dense_cm), dec).tolist()
        recall_per_class = np.round(PixelMetric.compute_recall_per_class(dense_cm), dec).tolist()

        for idx in self.ignore_labels:
            iou_per_class.pop(idx)
            F1_per_class.pop(idx)
            precision_per_class.pop(idx)
            recall_per_class.pop(idx)
            if self._class_names:
                self._class_names.pop(idx)

        mrecall = np.round(np.array(recall_per_class).mean(), dec)
        miou = np.round(np.array(iou_per_class).mean(), dec)
        mF1 = np.round(np.array(F1_per_class).mean(), dec)
        mprec = np.round(np.array(precision_per_class).mean(), dec)

        tb = pt.PrettyTable()
        if self._class_names:
            tb.field_names = ['name', 'class', 'iou', 'f1', 'precision', 'recall']
            for idx, (iou, f1, precision, recall) in enumerate(
                    zip(iou_per_class, F1_per_class, precision_per_class, recall_per_class)):
                tb.add_row([self._class_names[idx], idx, iou, f1, precision, recall])

            tb.add_row(['', 'mean', miou, mF1, mprec, mrecall])
        else:
            tb.field_names = ['class', 'iou', 'f1', 'precision', 'recall']
            for idx, (iou, f1, precision, recall) in enumerate(
                    zip(iou_per_class, F1_per_class, precision_per_class, recall_per_class)):
                tb.add_row([idx, iou, f1, precision, recall])

            tb.add_row(['mean', miou, mF1, mprec, mrecall])

        self._log_summary(tb, dense_cm)

        return tb, miou
