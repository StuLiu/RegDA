"""
@Filename:
@Project : Unsupervised_Domian_Adaptation
@date    : 2023-03-16 21:55
@Author  : WangLiu
@E-mail  : liuwa@hnu.edu.cn
"""
import os
import time
import torch
import argparse
import os.path as osp
import torch.backends.cudnn as cudnn
import torch.optim as optim

from regda.utils.eval import evaluate

from tqdm import tqdm
from torch.nn.utils import clip_grad
from ever.core.iterator import Iterator
from regda.datasets import *
from regda.gast.alignment import Aligner
from regda.gast.balance import *
from regda.utils.tools import *
from regda.models.Encoder import Deeplabv2
from regda.datasets.daLoader import DALoader
from regda.loss import PrototypeContrastiveLoss
from regda.gast.pseudo_generation import pseudo_selection

# CUDA_VISIBLE_DEVICES=7 python tools/init_prototypes.py --config-path st.proca.2potsdam --ckpt-model
# log/GAST/2potsdam/src/Potsdam_best.pth --ckpt-proto log/proca/2potsdam/align/prototypes_best.pth

parser = argparse.ArgumentParser(description='init proto')

parser.add_argument('--config-path', type=str, default='st.proca.2vaihingen', help='config path')

parser.add_argument('--ckpt-model', type=str,
                    default='log/proca/2vaihingen/src/Vaihingen_best.pth', help='model ckpt from stage1')

parser.add_argument('--ckpt-proto', type=str,
                    default='log/proca/2vaihingen/src/prototypes_best.pth', help='prototypes ckpt from stage1')

parser.add_argument('--stage', type=int, default=1, help='stage1')
args = parser.parse_args()

# get config from config.py
cfg = import_config(args.config_path, create=True, copy=False, postfix='/src' if args.stage == 1 else '/align')


def main():
    time_from = time.time()

    logger = get_console_file_logger(name=args.config_path.split('.')[1], logdir=cfg.SNAPSHOT_DIR)
    logger.info(os.path.basename(__file__))
    logging_args(args, logger)
    logging_cfg(cfg, logger)

    ignore_label = eval(cfg.DATASETS).IGNORE_LABEL
    class_num = len(eval(cfg.DATASETS).LABEL_MAP)
    model_name = str(cfg.MODEL).lower()
    # stop_steps = cfg.STAGE2_STEPS

    if model_name == 'resnet':
        model_name = 'resnet50'
    logger.info(model_name)

    cudnn.enabled = True
    model = Deeplabv2(dict(
        backbone=dict(
            resnet_type=model_name,
            output_stride=16,
            pretrained=True,
        ),
        multi_layer=True,
        cascade=False,
        use_ppm=True,
        ppm=dict(
            num_classes=class_num,
            use_aux=False,
            fc_dim=2048,
        ),
        inchannels=2048,
        num_classes=class_num,
        is_ins_norm=True,
    ))

    ckpt_model = torch.load(args.ckpt_model, map_location=torch.device('cpu'))
    model.load_state_dict(ckpt_model)
    model = model.cuda()

    aligner = Aligner(logger=logger,
                      feat_channels=2048,
                      class_num=class_num,
                      ignore_label=ignore_label,
                      decay=0.996)

    # source and target loader
    sourceloader = DALoader(cfg.SOURCE_DATA_CONFIG, cfg.DATASETS)
    sourceloader_iter = Iterator(sourceloader)

    for _ in tqdm(range(len(sourceloader))):
        # source infer
        batch = sourceloader_iter.next()
        images_s, label_s = batch[0]
        images_s, label_s = images_s.cuda(), label_s['cls'].cuda()
        pred_s1, pred_s2, feat_s = model(images_s)

        # avg-updating prototypes
        aligner.update_avg(feat_s, label_s)

    aligner.init_avg()
    torch.save(aligner.prototypes.cpu(), args.ckpt_proto)
    logger.info(f'>>>> Usning {float(time.time() - time_from) / 3600:.3f} hours.')


if __name__ == '__main__':
    # seed_torch(int(time.time()) % 10000019)
    seed_torch(2333)
    main()
