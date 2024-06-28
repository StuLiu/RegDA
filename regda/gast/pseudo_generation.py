"""
@Project : rsda 
@File    : pseudo_generation.py
@IDE     : PyCharm 
@Author  : Wang Liu
@Date    : 2023/2/19 16:06
@e-mail  : liuwa@hnu.edu.cn
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as tnf
import math

from glob import glob
import matplotlib.pyplot as plt
from regda.utils.tools import *
from regda.viz import VisualizeSegmm
from regda.datasets import *
from regda.utils.tools import AverageMeter


def pseudo_selection1(mask, cutoff_top=0.8, cutoff_low=0.6, return_type='ndarray', ignore_label=-1):
    """
    Convert continuous mask into binary mask
    Args:
        mask: torch.Tensor, the predicted probabilities for each examples, shape=(b, c, h, w).
        cutoff_top: float, the ratio for computing threshold.
        cutoff_low: float, the minimum threshold.
        return_type: str, the format of the return item, should be in ['ndarray', 'tensor'].
    Returns:
        ret: Tensor, pseudo label, shape=(b, h, w).
    """
    assert return_type in ['ndarray', 'tensor']
    assert mask.max() <= 1 and mask.min() >= 0, print(mask.max(), mask.min())
    bs, c, h, w = mask.size()
    mask = mask.view(bs, c, -1)                                     # (b, c, h*w)

    # for each class extract the threshold confidence
    class_threshold = mask.max(-1, keepdim=True)[0] * cutoff_top    # (b, c, 1)
    min_threshold = cutoff_low * torch.ones_like(class_threshold)   # (b, c, 1)
    class_threshold = class_threshold.max(min_threshold)            # (b, c, 1)
    class_threshold = class_threshold.permute(0, 2, 1)              # (b, 1, c)

    # get the max class confidence and score
    probs, pseudo_label = torch.max(mask, dim=1)                    # (b, h*w)
    pseudo_label_onehot = tnf.one_hot(pseudo_label, num_classes=c)  # (b, h*w, c)
    pixel_threshold = torch.sum(class_threshold * pseudo_label_onehot, dim=-1)      # (b, h*w)
    # if the top score is too low, ignore it
    pseudo_label[probs < pixel_threshold] = ignore_label             # (b, h*w)
    if return_type == 'ndarray':
        ret = pseudo_label.view(bs, h, w).cpu().numpy()
    else:
        ret = pseudo_label.view(bs, h, w)
    return ret


def pseudo_selection(mask, cutoff_top=0.8, cutoff_low=0.6, return_type='ndarray', ignore_label=-1):
    """
    Convert continuous mask into binary mask
    Args:
        mask: torch.Tensor, the predicted probabilities for each examples, shape=(b, c, h, w).
        cutoff_top: float, the ratio for computing threshold.
        cutoff_low: float, the minimum threshold.
        return_type: str, the format of the return item, should be in ['ndarray', 'tensor'].
    Returns:
        ret: Tensor, pseudo label, shape=(b, h, w).
    """
    assert return_type in ['ndarray', 'tensor']
    assert mask.max() <= 1 and mask.min() >= 0, print(mask.max(), mask.min())
    bs, c, h, w = mask.size()
    mask = mask.view(bs, c, -1)

    # for each class extract the max confidence
    mask_max, _ = mask.max(-1, keepdim=True)  # (b, c, 1)
    mask_max *= cutoff_top

    # if the top score is too low, ignore it
    lowest = torch.Tensor([cutoff_low]).type_as(mask_max)
    mask_max = mask_max.max(lowest)

    pseudo_gt = (mask > mask_max).type_as(mask)
    # remove ambiguous pixels, ambiguous = 1 means ignore
    ambiguous = (pseudo_gt.sum(1, keepdim=True) != 1).type_as(mask)

    pseudo_gt = pseudo_gt.argmax(dim=1, keepdim=True)
    pseudo_gt[ambiguous == 1] = ignore_label
    if return_type == 'ndarray':
        ret = pseudo_gt.view(bs, h, w).cpu().numpy()
    else:
        ret = pseudo_gt.view(bs, h, w)
    return ret


def gener_target_pseudo(_cfg, model, pseudo_loader, save_pseudo_label_path,
                        slide=True, save_prob=False, size=(1024, 1024), ignore_label=-1):
    """
    Generate pseudo label for target domain. The saved probabilities should be processed by softmax.
    Args:
        _cfg: cfg loaded from configs/
        model: nn.Module, deeplabv2
        pseudo_loader: DataLoader, dataloader for target domain image, batch_size=1
        save_pseudo_label_path: os.path.Path or str, the path for pseudo labels
        slide: bool, if use slide mode when do inferring.
        save_prob: bool, if save probabilities or class ids.
        size: tuple or list, the height and width of the pseudo labels.

    Returns:
        None
    """
    model.eval()

    save_pseudo_color_path = save_pseudo_label_path + '_color'
    os.makedirs(save_pseudo_label_path, exist_ok=True)
    os.makedirs(save_pseudo_color_path, exist_ok=True)
    viz_op = VisualizeSegmm(save_pseudo_color_path, eval(_cfg.DATASETS).PALETTE)
    num_classes = len(eval(_cfg.DATASETS).LABEL_MAP)

    with torch.no_grad():
        _i = 0
        for ret, ret_gt in tqdm(pseudo_loader):
            # _i += 1
            # if _i >= 1:
            #     break

            ret = ret.cuda()
            cls = pre_slide(model, ret, num_classes=num_classes, tta=True) if slide else model(ret)  # (b, c, h, w)

            if save_prob:
                # np.save(save_pseudo_label_path + '/' + ret_gt['fname'][0] + '.npy',
                #         tnf.interpolate(cls, size, mode='bilinear', align_corners=True).squeeze(
                #             dim=0
                #         ).cpu().numpy())       # (c, h, w)
                torch.save(tnf.interpolate(cls, size, mode='bilinear', align_corners=True).squeeze(dim=0).cpu(),
                           save_pseudo_label_path + '/' + ret_gt['fname'][0] + '.pt')   # (c, h, w)
                if _cfg.SNAPSHOT_DIR is not None:
                    cls = pseudo_selection(cls, ignore_label=ignore_label,
                                           cutoff_top=_cfg.CUTOFF_TOP, cutoff_low=_cfg.CUTOFF_LOW)  # (b, h, w)
                    for fname, pred in zip(ret_gt['fname'], cls):
                        viz_op(pred, fname.replace('.tif', '.png'))
            else:
                # pseudo selection, from -1~6
                if _cfg.PSEUDO_SELECT:
                    cls = pseudo_selection(cls, ignore_label=ignore_label)  # (b, h, w)
                    # cls = lbl.cpu().numpy()  # (b, h, w)
                else:
                    cls = cls.argmax(dim=1).cpu().numpy()

                cv2.imwrite(save_pseudo_label_path + '/' + ret_gt['fname'][0],
                            (cls + 1).reshape(*size).astype(np.uint8))

                if _cfg.SNAPSHOT_DIR is not None:
                    for fname, pred in zip(ret_gt['fname'], cls):
                        viz_op(pred, fname.replace('.tif', '.png'))


def analysis_pseudo_labels(label_dir='data/IsprsDA/Vaihingen/ann_dir/train',
                           pseudo_dir='log/GAST/2vaihingen/pseudo_label_1000',
                           ignore_label=-1, n_classes=6):

    labels = glob(label_dir + r'/*.png')
    pseudos = glob(pseudo_dir + r'/*.pt')
    assert len(labels) == len(pseudos)
    labels.sort()
    pseudos.sort()
    range_cnt = 100
    step_length = math.log(n_classes) / range_cnt
    cnt_true_list = np.zeros((range_cnt))   # for pl
    cnt_used_list = np.zeros((range_cnt))   # for pl
    acc_list = np.zeros((range_cnt))        # for pl
    diffi_list = np.zeros((range_cnt))      # for ohem
    acc_cnt, diffi_cnt = np.zeros((range_cnt)), np.zeros((range_cnt))
    # for i in tqdm(range(len(labels)//10)):
    for i in tqdm(range(len(labels))):
        lbl = cv2.imread(labels[i], cv2.IMREAD_UNCHANGED)
        lbl = torch.from_numpy(lbl).unsqueeze(0).to(torch.long).cuda()          # (1, h, w)
        gt = lbl.detach().clone()                                               # (1, h, w)
        cls = torch.load(pseudos[i], map_location=torch.device('cpu')
                         ).unsqueeze(0).cuda()                                  # (1, c, h, w)
        pseudo = pseudo_selection(cls, cutoff_top=0.8, cutoff_low=0.6,
                                  return_type='tensor', ignore_label=-1)        # (1, h, w)
        pseudo[pseudo == ignore_label] = n_classes                              # (1, h, w)
        # entropy
        entropy = torch.sum(-cls * torch.log(cls), dim=1)                       # (1, h, w)
        # entropy[pseudo == n_classes] = math.log(n_classes) + 1
        # difficulty
        lbl[lbl == ignore_label] = n_classes
        lbl_onehot = tnf.one_hot(lbl, num_classes=n_classes + 1)[:,:,:,:-1]     # (1, h, w, c)
        lbl_onehot = lbl_onehot.permute(0, 3, 1, 2)                             # (1, c, h, w)
        difficulty = 1 - torch.sum(cls * lbl_onehot, dim=1)                     # (1, h, w)
        # difficulty = torch.sum(-1 * lbl_onehot * torch.log(cls), dim=1)
        assert torch.sum(difficulty < 0) == 0
        # difficulty[pseudo == n_classes] = 100.0
        for i in range(range_cnt):
            v_fr = 1.0 * i * step_length
            v_to = v_fr + step_length
            cnt_true, cnt_used, acc_local, diffi_local = range_static(entropy, difficulty, pseudo, gt, v_fr, v_to,
                                                                      n_classes)
            cnt_used_list[i] = cnt_used_list[i] + cnt_used
            cnt_true_list[i] = cnt_true_list[i] + cnt_true
            acc_list[i] =  acc_list[i] + acc_local
            diffi_list[i] = diffi_list[i] + diffi_local
            if cnt_used != 0:
                acc_cnt[i] += 1
            if diffi_local != 0:
                diffi_cnt[i] += 1

    acc_list = [acc_list[i] / (acc_cnt[i] + 1e-7) for i in range(range_cnt)]
    diffi_list = [diffi_list[i] / (diffi_cnt[i] + 1e-7) for i in range(range_cnt)]
    x = [(i * step_length) for i in range(range_cnt)]
    # acc_list2 = [cnt_true_list[i] / (cnt_used_list[i] + 1e-7) for i in range(range_cnt)]
    # print(cnt_true_list)
    # print(cnt_used_list)
    # print(acc_list)
    # cnt_true_list = np.where(cnt_true_list > 1, np.log10(cnt_true_list), 0)
    # cnt_used_list = np.where(cnt_used_list > 1, np.log10(cnt_used_list), 0)

    show_tradeoff(x[:range_cnt//2], diffi_list[:range_cnt//2], cnt_used_list[:range_cnt//2])
    plot_noise_rate(x, acc_list, diffi_list, n_classes)
    plot_cnt(x, cnt_true_list, cnt_used_list)


def range_static(entropy, difficulty, pseudo, gt, v_fr=0.0, v_to=1.0, n_classes=6):
    pseudo_range = pseudo.detach().clone()
    pseudo_range[(entropy < v_fr) | (entropy >= v_to)] = n_classes
    cnt_true = torch.sum(pseudo_range == gt)
    cnt_used = torch.sum(pseudo_range != n_classes)
    acc = 1.0 * cnt_true / (cnt_used + 1e-7)

    difficulty_range = difficulty.detach().clone()
    difficulty_range[(entropy < v_fr) | (entropy >= v_to)] = 0
    diffi = 1.0 * torch.sum(difficulty_range) / (torch.sum((entropy >= v_fr) & (entropy < v_to)) + 1e-7)
    # print(cnt_used, cnt_true)
    return cnt_true, cnt_used, acc, diffi


def plot_noise_rate(x, acc_list, diffi_list, num_classes=6, block=False):
    # x = np.arange(1, 10, 1)  # 从1到9，间隔1取点

    # plt.plot(x, acc_list, "ob-", label="accuracy")  # "b"为蓝色, "o"为圆点, ":"为点线
    # plt.plot(x, diffi_list, "rs-", label="difficulty")  # "r"为红色, "s"为方块, "-"为实线
    #
    # plt.title('title')  # 标题 只能是英文
    # plt.xlabel("entropy")  # x轴名称 只能是英文
    # plt.ylabel("accuracy")  # y轴名称 只能是英文

    fig = plt.figure()

    # ax1显示y1  ,ax2显示y2
    ax1 = fig.subplots()
    ax2 = ax1.twinx()  # 使用twinx()，得到与ax1 对称的ax2,共用一个x轴，y轴对称（坐标不对称）
    ax1.plot(x, acc_list, 'b-', label="Accuracy", linewidth=3)
    ax2.plot(x, diffi_list, 'r-', label="Difficulty", linewidth=3)

    ax1.set_xlabel('Uncertainty')
    ax1.set_ylabel('Accuracy')
    ax2.set_ylabel('Difficulty')

    ax1.set_ylim(0, 1.0)
    ax2.set_ylim(0, 1.0)

    plt.xlim(0, math.log(num_classes))  # 限制x坐标轴范围
    # plt.xlim(0, 1.2)  # 限制x坐标轴范围

    lines = []
    labels = []
    for ax in fig.axes:
        axLine, axLabel = ax.get_legend_handles_labels()
        lines.extend(axLine)
        labels.extend(axLabel)

    # fig.legend(lines, labels, loc='upper center')

    # plt.grid(b=True, axis='y')  # 显示网格线

    plt.show(block=block)
    print('showed')


def plot_cnt(x, y1, y2):
    mycolors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:brown', 'tab:grey', 'tab:pink', 'tab:olive']
    columns = ['psavert', 'uempmed']

    # Draw Plot
    fig, ax = plt.subplots(1, 1, figsize=(16, 9), dpi=80)
    ax.fill_between(x, y1=y1, y2=0, label=columns[1], alpha=0.5, color=mycolors[0], linewidth=2)
    ax.fill_between(x, y1=y2, y2=0, label=columns[1], alpha=0.5, color=mycolors[1], linewidth=2)

    # Decorations
    ax.set_title('Personal Savings Rate vs Median Duration of Unemployment', fontsize=18)
    # ax.set(ylim=[0, 30])
    ax.legend(loc='best', fontsize=12)
    # plt.xticks(x[::50], fontsize=10, horizontalalignment='center')
    # plt.yticks(np.arange(2.5, 30.0, 2.5), fontsize=10)
    # plt.xlim(-10, x[-1])

    # # Draw Tick lines
    # for y in np.arange(0, 1.1, 0.2):
    #     plt.hlines(y, xmin=0, xmax=math.log(num_classes), colors='black', alpha=0.3, linestyles="--", lw=0.5)

    # Lighten borders
    plt.gca().spines["top"].set_alpha(0)
    plt.gca().spines["bottom"].set_alpha(.3)
    plt.gca().spines["right"].set_alpha(0)
    plt.gca().spines["left"].set_alpha(.3)
    plt.show()


def show_tradeoff(unce, diffi, cnt_used_list, t=0.75, gamma=1, block=False):
    unce_ = torch.from_numpy(np.array(unce))
    diffi_ = torch.from_numpy(np.array(diffi))

    def get_weight(uncertainties, gamma=8 ,m=0.1, t=0.75):
        weight_left = -1 / m ** 2 * (uncertainties - m) ** 2 + 1
        weight_right = -1 / (t - m) ** 2 * (uncertainties - m) ** 2 + 1
        weight = torch.where(uncertainties < m, weight_left ** (1.0/gamma), weight_right ** (1.0/gamma))
        return weight

    weight_hard = ((1 - torch.exp(-diffi_)) ** gamma).detach()
    # weight_unce = ((1 - unce_**2 / t**2)).detach()
    weight_unce = get_weight(unce_)
    weight_hard = weight_hard / (weight_hard.max() + 1e-7)
    # weight_unce = weight_unce / (weight_unce.max() + 1e-7)
    plt.plot(unce, weight_hard, "ob-", label="difficulty")  # "b"为蓝色, "o"为圆点, ":"为点线
    plt.plot(unce, weight_unce, "rs-", label="uncertainty")  # "r"为红色, "s"为方块, "-"为实线
    # plt.show(block=block)
    weight = weight_hard * weight_unce
    # weight_right = weight_hard * weight_unce
    # weight = torch.where(unce_ <= t / 2, weight_left, weight_right)
    weight = weight / (weight.max() + 1e-7) * 0.5 + 0.5
    # plt.plot(unce, (weight_hard + weight_unce) / 2, "rb-", label="difficulty")  # "b"为蓝色, "o"为圆点, ":"为点线
    plt.plot(unce, weight, "go-", label="uncertainty")  # "r"为红色, "s"为方块, "-"为实线
    plt.show(block=block)
    print('showed')


if __name__ == "__main__":
    analysis_pseudo_labels(label_dir='data/IsprsDA/Vaihingen/ann_dir/train',
                           pseudo_dir='log/GAST/2vaihingen/pseudo_label',
                           ignore_label=-1, n_classes=6)

    # analysis_pseudo_labels(label_dir='data/IsprsDA/Potsdam/ann_dir/train',
    #                        pseudo_dir='log/GAST/2potsdam20230910153254/pseudo_label',
    #                        ignore_label=-1, n_classes=6)

