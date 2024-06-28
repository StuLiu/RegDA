import importlib
import logging
import time
import os
import shutil
import random
import torch
import torch.nn as nn
import torch.nn.functional as tnf
import numpy as np
import ttach
import ever as er
import argparse
# import pydensecrf.densecrf as dcrf # require py36

from skimage.io import imsave
from functools import reduce
from collections import OrderedDict
from tqdm import tqdm
from math import *
from scipy import ndimage


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


def logging_args(args_namespace, logger):
    logger.info(f'>>>>>>>>>>>>>>>>>>>>> arguments logging begin:')
    for k, v in vars(args_namespace).items():
        logger.info(f'{k}={v}')
    logger.info(f'<<<<<<<<<<<<<<<<<<<<< arguments logging end!')


def logging_cfg(cfg, logger):
    logger.info(f'>>>>>>>>>>>>>>>>>>>>> config logging begin:')
    logger.info(cfg.__name__)
    for k, v in vars(cfg).items():
        if str(k)[:2] != '__':
            logger.info(f'{k}={v}')
    logger.info(f'<<<<<<<<<<<<<<<<<<<<< config logging end!')


def get_curr_time():
    return f'{time.strftime("%Y%m%d%H%M%S", time.localtime())}'


def pad_image(img, target_size):
    """Pad an image up to the target size."""
    rows_missing = target_size[0] - img.shape[2]
    cols_missing = target_size[1] - img.shape[3]
    padded_img = tnf.pad(img, (0, 0, rows_missing, cols_missing), 'constant', 0)
    return padded_img


def pre_slide(model, image, num_classes=7, tile_size=(512, 512), tta=False):
    image_size = image.shape  # i.e. (1,3,1024,1024)
    overlap = 1 / 2  # 每次滑动的重合率为1/2

    stride = ceil(tile_size[0] * (1 - overlap))  # 滑动步长:512*(1-1/2) = 256
    tile_rows = int(ceil((image_size[2] - tile_size[0]) / stride) + 1)  # 行滑动步数:(1024-512)/256 + 1 = 3
    tile_cols = int(ceil((image_size[3] - tile_size[1]) / stride) + 1)  # 列滑动步数:(1024-512)/256 + 1 = 3

    full_probs = torch.zeros((image_size[0], num_classes, image_size[2], image_size[3])).cuda()  # 初始化全概率矩阵 (1,7,1024,1024)
    count_predictions = torch.zeros((image_size[0], 1, image_size[2], image_size[3])).cuda()  # 初始化计数矩阵 (1,1,1024,1024)

    for row in range(tile_rows):  # row = 0,1,2
        for col in range(tile_cols):  # col = 0,1,2
            x1 = int(col * stride)  # 起始位置x1 = 0 * 256 = 0
            y1 = int(row * stride)  # y1 = 0 * 256 = 0
            x2 = min(x1 + tile_size[1], image_size[3])  # 末位置x2 = min(0+512, 1024)
            y2 = min(y1 + tile_size[0], image_size[2])  # y2 = min(0+512, 1024)
            x1 = max(int(x2 - tile_size[1]), 0)  # 重新校准起始位置x1 = max(512-512, 0)
            y1 = max(int(y2 - tile_size[0]), 0)  # y1 = max(512-512, 0)

            img = image[:, :, y1:y2, x1:x2]  # 滑动窗口对应的图像 imge[:, :, 0:512, 0:512]
            padded_img = pad_image(img, tile_size)  # padding 确保扣下来的图像为512*512

            # 将扣下来的部分传入网络，网络输出概率图。
            # use softmax
            if tta is True:
                padded = tta_predict(model, padded_img)
            else:
                padded = model(padded_img)
                # padded = tnf.softmax(padded, dim=1)

            pre = padded[:, :, 0:img.shape[2], 0:img.shape[3]]  # 扣下相应面积 shape(1,7,512,512)
            count_predictions[:, :, y1:y2, x1:x2] += 1  # 窗口区域内的计数矩阵加1
            full_probs[:, :, y1:y2, x1:x2] += pre  # 窗口区域内的全概率矩阵叠加预测结果
    # average the predictions in the overlapping regions
    full_probs /= count_predictions
    return full_probs  # 返回整张图的平均概率 shape(1, 1, 1024,1024)


def predict_whole(model, image, tile_size):
    image = torch.from_numpy(image)
    interp = nn.Upsample(size=tile_size, mode='bilinear', align_corners=True)
    x = model(image.cuda())
    x = interp(x)
    return x


def predict_multiscale(model, image, scales=(0.75, 1.0, 1.25, 1.5, 1.75, 2.0), tile_size=(512, 512)):
    """
    Predict an image by looking at it with different scales.
        We choose the "predict_whole_img" for the image with less than the original input size,
        for the input of larger size, we would choose the cropping method to ensure that GPU memory is enough.
    """
    image_size = image.shape
    image = image.data.cpu().numpy()

    full_probs = torch.zeros((1, 1, image_size[2], image_size[3])).cuda()  # 初始化全概率矩阵 shape(1024,2048,19)

    for scale in scales:
        scale = float(scale)
        print("Predicting image scaled by %f" % scale)
        scale_image = ndimage.zoom(image, (1.0, 1.0, scale, scale), order=1, prefilter=False)

        scaled_probs = predict_whole(model, scale_image, tile_size)
        full_probs += scaled_probs

    full_probs /= len(scales)

    return full_probs


def tta_predict(model, img):
    tta_transforms = ttach.Compose(
        [
            ttach.HorizontalFlip(),
            ttach.Rotate90(angles=[0, 90, 180, 270]),
        ])

    xs = []

    for t in tta_transforms:
        aug_img = t.augment_image(img)
        aug_x = model(aug_img)
        # aug_x = tnf.softmax(aug_x, dim=1)

        x = t.deaugment_mask(aug_x)
        xs.append(x)

    xs = torch.cat(xs, 0)
    x = torch.mean(xs, dim=0, keepdim=True)

    return x


def mixup(s_img, s_lab, t_img, t_lab):
    s_lab, t_lab = s_lab.unsqueeze(1), t_lab.unsqueeze(1)

    batch_size = s_img.size(0)
    rand = torch.randperm(batch_size)
    lam = int(np.random.beta(0.2, 0.2) * s_img.size(2))

    new_s_img = torch.cat([s_img[:, :, 0:lam, :], t_img[rand][:, :, lam:s_img.size(2), :]], dim=2)
    new_s_lab = torch.cat([s_lab[:, :, 0:lam, :], t_lab[rand][:, :, lam:s_img.size(2), :]], dim=2)

    new_t_img = torch.cat([t_img[rand][:, :, 0:lam, :], s_img[:, :, lam:t_img.size(2), :]], dim=2)
    new_t_lab = torch.cat([t_lab[rand][:, :, 0:lam, :], s_lab[:, :, lam:t_img.size(2), :]], dim=2)

    new_s_lab, new_t_lab = new_s_lab.squeeze(1), new_t_lab.squeeze(1)

    return new_s_img, new_s_lab, new_t_img, new_t_lab


def import_config(config_name, prefix='configs', copy=True, create=True, postfix=''):
    cfg_path = '{}.{}'.format(prefix, config_name)
    m = importlib.import_module(name=cfg_path)
    m.SNAPSHOT_DIR += postfix
    if create:
        os.makedirs(m.SNAPSHOT_DIR, exist_ok=True)
    if copy:
        shutil.copy(cfg_path.replace('.', '/') + '.py', os.path.join(m.SNAPSHOT_DIR, 'config.py'))
    return m


def portion_warmup(i_iter, start_iter, end_iter):
    if i_iter < start_iter or i_iter > end_iter or start_iter >= end_iter:
        return 0
    return 2.0 / (1.0 + exp(-10 * float(i_iter - start_iter) / float(end_iter - start_iter))) - 1
    # return float(i_iter - start_iter) / float(end_iter - start_iter)


def lr_poly(base_lr, i_iter, max_iter, power):
    return base_lr * ((1 - float(i_iter) / max_iter) ** power)


def lr_warmup(base_lr, i_iter, warmup_iter):
    return base_lr * (float(i_iter) / warmup_iter)


def adjust_learning_rate(optimizer, i_iter, cfg):
    if i_iter < cfg.PREHEAT_STEPS:
        lr = lr_warmup(cfg.LEARNING_RATE, i_iter, cfg.PREHEAT_STEPS)
    else:
        lr = lr_poly(cfg.LEARNING_RATE, i_iter, cfg.NUM_STEPS, cfg.POWER)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10
    return lr


def adjust_learning_rate_D(optimizer, i_iter, cfg):
    if i_iter < cfg.PREHEAT_STEPS:
        lr = lr_warmup(cfg.LEARNING_RATE_D, i_iter, cfg.PREHEAT_STEPS)
    else:
        lr = lr_poly(cfg.LEARNING_RATE_D, i_iter, cfg.NUM_STEPS, cfg.POWER)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10
    return lr


def get_console_file_logger(name, level=logging.INFO, logdir='./baseline'):
    logger = logging.Logger(name)
    logger.setLevel(level=level)
    logger.handlers = []
    basic_format = "%(asctime)s, %(levelname)s:%(name)s:%(message)s"
    date_format = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(basic_format, date_format)
    chlr = logging.StreamHandler()
    chlr.setFormatter(formatter)
    chlr.setLevel(level=level)

    fhlr = logging.FileHandler(os.path.join(logdir, str(time.time()) + '.log'))
    fhlr.setFormatter(formatter)
    logger.addHandler(chlr)
    logger.addHandler(fhlr)

    return logger


def loss_calc(pred, label, loss_fn, multi=False):
    """
    This function returns cross entropy loss for semantic segmentation
    """

    if multi is True:
        loss = 0
        num = 0
        for p in pred:
            if p.size()[-2:] != label.size()[-2:]:
                p = tnf.interpolate(p, size=label.size()[-2:], mode='bilinear', align_corners=True)
            # l = tnf.cross_entropy(p, label.long(), ignore_index=-1, reduction=reduction)
            loss += loss_fn(p, label.long())
            num += 1
        loss = loss / num
    else:
        if pred.size()[-2:] != label.size()[-2:]:
            pred = tnf.interpolate(pred, size=label.size()[-2:], mode='bilinear', align_corners=True)
        loss = loss_fn(pred, label.long())

    return loss


def bce_loss(pred, label):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    return tnf.binary_cross_entropy_with_logits(pred, label)


def robust_binary_crossentropy(pred, tgt):
    inv_tgt = -tgt + 1.0
    inv_pred = -pred + 1.0 + 1e-6
    return -(tgt * torch.log(pred + 1e-6) + inv_tgt * torch.log(inv_pred))


def bugged_cls_bal_bce(pred, tgt):
    inv_tgt = -tgt + 1.0
    inv_pred = pred + 1.0 + 1e-6
    return -(tgt * torch.log(pred + 1.0e-6) + inv_tgt * torch.log(inv_pred))


def adjust_confidence(i_iter, max_iter, cfg):
    confi_max, confi_min = cfg['confidence_maxin']
    if cfg['schedule'] == 'ploy':
        confi = (confi_max - confi_min) * ((1 - float(i_iter) / max_iter) ** (cfg['power'])) + confi_min
    else:
        confi = confi_min
    return confi


def som(loss, ratio=0.5, reduction='none'):
    # 1. keep num
    num_inst = loss.numel()
    num_hns = int(ratio * num_inst)
    # 2. select loss
    top_loss, _ = loss.reshape(-1).topk(num_hns, -1)
    if reduction == 'none':
        return top_loss
    else:
        loss_mask = (top_loss != 0)
        # 3. mean loss
        return torch.sum(top_loss[loss_mask]) / (loss_mask.sum() + 1e-6)


def seed_torch(seed=2333):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.enabled = True


def seed_worker():
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def ias_thresh(conf_dict, n_class, alpha, w=None, gamma=1.0):
    if w is None:
        w = np.ones(n_class)
    # threshold
    cls_thresh = np.ones(n_class, dtype=np.float32)
    for idx_cls in np.arange(0, n_class):
        if conf_dict[idx_cls] is not None:
            arr = np.array(conf_dict[idx_cls])
            cls_thresh[idx_cls] = np.percentile(arr, 100 * (1 - alpha * w[idx_cls] ** gamma))
    return cls_thresh


def generate_pseudo(model, target_loader, save_dir, n_class=7, pseudo_dict=None, logger=None):
    if pseudo_dict is None:
        pseudo_dict = dict()
    logger.info('Start generate pseudo labels: %s' % save_dir)
    viz_op = er.viz.VisualizeSegmm(os.path.join(save_dir, 'vis'), palette)
    os.makedirs(os.path.join(save_dir, 'pred'), exist_ok=True)
    model.eval()
    cls_thresh = np.ones(n_class) * 0.9
    for image, labels in tqdm(target_loader):
        out = model(image.cuda())
        logits = out[0] if isinstance(out, tuple) else out
        max_items = logits.max(dim=1)
        label_pred = max_items[1].data.cpu().numpy()
        logits_pred = max_items[0].data.cpu().numpy()

        logits_cls_dict = {c: [cls_thresh[c]] for c in range(n_class)}
        for cls in range(n_class):
            logits_cls_dict[cls].extend(logits_pred[label_pred == cls].astype(np.float16))
        # instance adaptive selector
        tmp_cls_thresh = ias_thresh(logits_cls_dict, n_class, pseudo_dict['pl_alpha'], w=cls_thresh,
                                    gamma=pseudo_dict['pl_gamma'])
        beta = pseudo_dict['pl_beta']
        cls_thresh = beta * cls_thresh + (1 - beta) * tmp_cls_thresh
        cls_thresh[cls_thresh >= 1] = 0.999

        np_logits = logits.data.cpu().numpy()
        for _i, fname in enumerate(labels['fname']):
            # save pseudo label
            logit = np_logits[_i].transpose(1, 2, 0)
            label = np.argmax(logit, axis=2)
            logit_amax = np.amax(logit, axis=2)
            label_cls_thresh = np.apply_along_axis(lambda x: [cls_thresh[_e] for _e in x], 1, label)
            ignore_index = logit_amax < label_cls_thresh
            viz_op(label, fname)
            label += 1
            label[ignore_index] = 0
            imsave(os.path.join(save_dir, 'pred', fname), label.astype(np.uint8))

    return os.path.join(save_dir, 'pred')


def entropyloss(logits, weight=None):
    """
    logits:     N * C * H * W
    weight:     N * 1 * H * W
    """
    val_num = weight[weight > 0].numel()
    logits_log_softmax = torch.log_softmax(logits, dim=1)
    entropy = -torch.softmax(logits, dim=1) * weight * logits_log_softmax
    entropy_reg = torch.sum(entropy) / val_num
    return entropy_reg


def kldloss(logits, weight):
    """
    logits:     N * C * H * W
    weight:     N * 1 * H * W
    """
    val_num = weight[weight > 0].numel()
    logits_log_softmax = torch.log_softmax(logits, dim=1)
    num_classes = logits.size()[1]
    kld = - 1 / num_classes * weight * logits_log_softmax
    kld_reg = torch.sum(kld) / val_num
    return kld_reg


def count_model_parameters(module, _default_logger=None):
    cnt = 0
    for p in module.parameters():
        cnt += reduce(lambda x, y: x * y, list(p.shape))
    _default_logger.info('#params: {}, {} M'.format(cnt, round(cnt / float(1e6), 3)))

    return cnt


def index2onehot(labels, class_num=7, ignore_label=-1):
    if labels.dim() == 4:
        labels = labels.squeeze(dim=1)
    labels[labels == ignore_label] = class_num
    labels_onehot = tnf.one_hot(labels, num_classes=class_num + 1)[:, :, :, :-1]  # (b*h*w, c)
    labels_onehot = labels_onehot.permute(0, 3, 1, 2)
    return labels_onehot    # (b, c, h, w)


class BestLog:
    def __init__(self, high=True):
        self.value = -999999 if high else 999999
        self.iter = 0
        self.log_str = ''

    def update(self, val, it, log_str):
        cond = (val >= self.value) if self.high else (val <= self.value)
        if cond:
            self.iter = it
            self.log_str = log_str


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    seed_torch(2333)
    s = torch.randn((5, 5)).cuda()
    print(s)
    for i in range(1000):
        print(portion_warmup(i_iter=i, start_iter=0, end_iter=1000))

# def get_crf(mask, img, num_classes=7, size=512):
#     mask = np.transpose(mask, (2, 0, 1))
#     img = np.ascontiguousarray(img)
#
#     unary = -np.log(mask + 1e-8)
#     unary = unary.reshape((num_classes, -1))
#     unary = np.ascontiguousarray(unary)
#
#     d = dcrf.DenseCRF2D(size, size, num_classes)
#     d.setUnaryEnergy(unary)
#
#     d.addPairwiseGaussian(sxy=5, compat=3)
#     d.addPairwiseBilateral(sxy=10, srgb=13, rgbim=img, compat=10)
#
#     output = d.inference(10)
#     map = np.argmax(output, axis=0).reshape((size, size))
#     return map
#
#
# def crf_predict(model, img, size=1024):
#     output = model(img)
#
#     mask = tnf.softmax(output, dim=1)
#     mask = mask.squeeze(0).permute(1, 2, 0).cpu().numpy()
#     mask = mask.astype(np.float32)
#
#     img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
#
#     # img = Normalize_back(img, flag=opt.dataset)
#     crf_out = get_crf(mask, img.astype(np.uint8), size=size)
#     return crf_out
