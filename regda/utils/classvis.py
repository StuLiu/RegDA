"""
@Project : rads2
@File    : classvis.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2023/6/9 上午11:44
@e-mail  : 1183862787@qq.com
"""
import random
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import warnings; warnings.filterwarnings(action='always')
from regda.datasets import IsprsDA, LoveDA
from regda.datasets.daLoader import DALoader
from regda.utils.tools import AverageMeter
import glob
import cv2
from tqdm import tqdm


def mask_loader(mask_dir='../../data/IsprsDA/Potsdam/ann_dir/train', class_num=6, offset=-1):
    mask_paths = glob.glob(mask_dir + r'/*.png')
    mask_paths.sort()
    for mask_path in mask_paths:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) + offset
        ele_cnt = np.size(mask)
        local_ele = np.zeros([class_num])

        # print(np.unique(mask))
        for i in range(class_num):
            class_cnt = np.sum(mask == i)
            local_ele[i] = float(class_cnt) / float(ele_cnt)
        yield local_ele.astype(np.float32)


def DA_class_bar(datasets='isprsda', max_rate=0.5):
    assert datasets in ['isprsda', 'loveda']
    if datasets == 'isprsda':
        label_map = IsprsDA.LABEL_MAP
        x_value = list(label_map.values())
        x_class = list(label_map.keys())

        ele_rate = np.zeros([len(x_class)]).astype(np.float32)
        for local_ele in tqdm(mask_loader('../../data/IsprsDA/Potsdam/ann_dir/train', len(x_class))):
            ele_rate = ele_rate + local_ele
        val_1 = ele_rate / np.sum(ele_rate)

        ele_rate = np.zeros([len(x_class)]).astype(np.float32)
        for local_ele in tqdm(mask_loader('../../data/IsprsDA/Vaihingen/ann_dir/train', len(x_class))):
            ele_rate = ele_rate + local_ele
        val_2 = ele_rate / np.sum(ele_rate)
        x = np.array(x_value)

        domain1_name = 'Potsdam'
        domain2_name = 'Vaihingen'
    else:
        label_map = LoveDA.LABEL_MAP
        x_value = list(label_map.values())
        x_class = list(label_map.keys())

        ele_rate = np.zeros([len(x_class)]).astype(np.float32)
        for local_ele in tqdm(mask_loader('../../data/LoveDA/Train/Urban/masks_png', len(x_class))):
            ele_rate = ele_rate + local_ele
        val_1 = ele_rate / np.sum(ele_rate)

        ele_rate = np.zeros([len(x_class)]).astype(np.float32)
        for local_ele in tqdm(mask_loader('../../data/LoveDA/Train/Rural/masks_png', len(x_class))):
            ele_rate = ele_rate + local_ele
        val_2 = ele_rate / np.sum(ele_rate)
        x = np.array(x_value)

        domain1_name = 'Urban'
        domain2_name = 'Rural'

    total_width, n = 0.8, 2
    width = float(total_width) / float(n)
    plt.barh(x, val_1, height=width, tick_label=x_class, label=domain1_name)
    plt.barh(x + width, val_2, height=width, tick_label=x_class, label=domain2_name)

    for i in range(len(x)):
        plt.text(val_1[i] + 0.02, x[i] - 0.05, "%.2f" % val_1[i], ha='center', fontsize=10)
        plt.text(val_2[i] + 0.02, x[i] + width - 0.05, "%.2f" % val_2[i], ha='center', fontsize=10)

    plt.xlim(0, max_rate)
    plt.xlabel('ratio')
    plt.ylabel('class')
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.1)

    # plt.xticks(rotation=45, position=(0,0))
    # plt.bar(x + 2 * width, c, width=width, label='c')
    plt.legend()
    plt.show()


DA_class_bar(datasets='isprsda', max_rate=0.35)
DA_class_bar(datasets='loveda', max_rate=0.55)


# # set the  fontsize and some other elements
# large = 22
# med = 16
# small = 12
# params = {'axes.titlesize': large,
#           'legend.fontsize': med,
#           'figure.figsize': (16, 10),
#           'axes.labelsize': med,
#           'axes.titlesize': med,
#           'xtick.labelsize': med,
#           'ytick.labelsize': med,
#           'figure.titlesize': large}
# plt.rcParams.update(params)
# plt.style.use('seaborn-whitegrid')
# sns.set_style("white")
#
#
# # Print Version
# print(mpl.__version__)
# print(sns.__version__)
#
#
# # Import Data
# df_raw = pd.read_csv("mpg_ggplot2.csv")
#
# # Prepare Data
# df = df_raw.groupby('manufacturer').size().reset_index(name='counts')
# n = df['manufacturer'].unique().__len__()+1
# all_colors = list(plt.cm.colors.cnames.keys())
# random.seed(100)
# c = random.choices(all_colors, k=n)
#
# # Plot Bars
# plt.figure(figsize=(16,10), dpi= 80)
# plt.bar(df['manufacturer'], df['counts'], color=c, width=.5)
# for i, val in enumerate(df['counts'].values):
#     plt.text(i, val, float(val), horizontalalignment='center', verticalalignment='bottom', fontdict={'fontweight':500, 'size':12})
#
# # Decoration
# plt.gca().set_xticklabels(df['manufacturer'], rotation=60, horizontalalignment= 'right')
# plt.title("Number of Vehicles by Manaufacturers", fontsize=22)
# plt.ylabel('# Vehicles')
# plt.ylim(0, 45)
# plt.show()