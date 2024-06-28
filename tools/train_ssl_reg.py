"""
@Filename:
@Project : Unsupervised_Domian_Adaptation
@date    : 2023-03-16 21:55
@Author  : WangLiu
@E-mail  : liuwa@hnu.edu.cn
"""

import torch.multiprocessing

import os.path as osp
import torch.backends.cudnn as cudnn
import torch.optim as optim
from regda.utils.eval import evaluate
from regda.utils.tools import *
from regda.models.Encoder import Deeplabv2
from regda.datasets.daLoader import DALoader
from regda.datasets import *

from ever.core.iterator import Iterator
from tqdm import tqdm
from torch.nn.utils import clip_grad
from regda.gast.alignment import Aligner
from regda.gast.pseudo_generation import gener_target_pseudo, pseudo_selection
from regda.gast.balance import *
from regda.utils.ema import ExponentialMovingAverage
from regda.utils.local_region_homog import Homogenizer


# arg parser
# --config-path st.proca.2urban --refine-label 1 --refine-mode all --refine-temp 2 --balance-class 1 --balance-temp 0.5
# --config-path st.proca.2rural --refine-label 1 --refine-mode all --refine-temp 2 --balance-class 1 --balance-temp 1000

parser = argparse.ArgumentParser(description='Run proca methods. ssl')
parser.add_argument('--config-path', type=str, default='st.uvem.2vaihingen', help='config path')
# ckpts
parser.add_argument('--ckpt-model', type=str,
                    default='log/uvem/2vaihingen/align/Vaihingen_best.pth', help='model ckpt from stage1')
parser.add_argument('--ckpt-proto', type=str,
                    default='log/uvem/2vaihingen/align/prototypes_best.pth', help='proto ckpt from stage1')

parser.add_argument('--gen', type=str2bool, default=1, help='if generate pseudo-labels')
# MPC
parser.add_argument('--refine-label', type=str2bool, default=1, help='whether refine the pseudo label')
parser.add_argument('--refine-mode', type=str, default='all', choices=['all'],
                    help='refine by prototype, label, or both')
parser.add_argument('--refine-temp', type=float, default=2.0, help='whether refine the pseudo label')
# LRH
parser.add_argument('--sam-refine', action='store_true', help='whether LRH')
parser.add_argument('--percent', type=float, default=0.5, help='class-freq threshold for LRH')
# source loss
parser.add_argument('--ls', type=str, default="CrossEntropy",
                    choices=['CrossEntropy', 'OhemCrossEntropy'], help='source loss function')
parser.add_argument('--bcs', type=str2bool, default=0, help='whether balance class for source')
# target loss
parser.add_argument('--lt', type=str, default='none',
                    choices=['ours', 'uvem', 'ohem', 'focal', 'ghm', 'ups', 'none'], help='target loss function')
parser.add_argument('--bct', type=str2bool, default=0, help='whether balance class for target')
parser.add_argument('--class-temp', type=float, default=2.0, help='smooth factor')
# UVEM
parser.add_argument('--uvem-m', type=float, default=0.2, help='whether balance class')
parser.add_argument('--uvem-t', type=float, default=0.7, help='whether balance class')
parser.add_argument('--uvem-g', type=float, default=4, help='whether balance class')

args = parser.parse_args()

# get config from config.py
cfg = import_config(args.config_path, create=True, copy=True, postfix=f'/ssl')
print('args.sam_refine,', args.sam_refine)


def main():
    time_from = time.time()
    save_pseudo_label_path = osp.join(cfg.SNAPSHOT_DIR, 'pseudo_label')
    os.makedirs(save_pseudo_label_path, exist_ok=True)

    logger = get_console_file_logger(name=args.config_path.split('.')[1], logdir=cfg.SNAPSHOT_DIR)
    logger.info(os.path.basename(__file__))
    logging_args(args, logger)
    logging_cfg(cfg, logger)

    ignore_label = eval(cfg.DATASETS).IGNORE_LABEL
    class_num = len(eval(cfg.DATASETS).LABEL_MAP)
    model_name = str(cfg.MODEL).lower()
    stop_steps = cfg.STAGE3_STEPS
    cfg.NUM_STEPS = stop_steps * 1.5            # for learning rate poly
    cfg.PREHEAT_STEPS = int(stop_steps / 20)    # for warm-up

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
                      decay=0.996,
                      resume=args.ckpt_proto)

    homogenizer = Homogenizer(percent=args.percent, class_num=class_num, ignore_label=ignore_label)

    class_balancer_s = ClassBalance(class_num=class_num,
                                    ignore_label=ignore_label,
                                    decay=0.99,
                                    temperature=args.class_temp)
    class_balancer_t = ClassBalance(class_num=class_num,
                                    ignore_label=ignore_label,
                                    decay=0.99,
                                    temperature=args.class_temp)

    loss_fn_s = eval(args.ls)(ignore_label=ignore_label, class_balancer=class_balancer_s if args.bcs else None)
    if args.lt in ['ours', 'uvem']:
        logger.info('>>>>>>> using ours/uvem_loss.')
        loss_fn_t = UVEMLoss(m=args.uvem_m,
                             threshold=args.uvem_t,
                             gamma=args.uvem_g,
                             class_balancer=class_balancer_t if args.bct else None,
                             class_num=class_num,
                             ignore_label=ignore_label)
    elif args.lt == 'ohem':
        logger.info('>>>>>>> using OhemCrossEntropy.')
        loss_fn_t = OhemCrossEntropy(ignore_label=ignore_label)
    elif args.lt == 'focal':
        logger.info('>>>>>>> using FocalLoss.')
        loss_fn_t = FocalLoss(gamma=2.0, reduction='mean', ignore_label=ignore_label)
    elif args.lt == 'ghm':
        logger.info('>>>>>>> using GHMLoss.')
        loss_fn_t = GHMLoss(bins=30, momentum=0.99, ignore_label=ignore_label)
    elif args.lt == 'ups':
        logger.info('>>>>>>> using UPSLoss.')
        loss_fn_t = UPSLoss(threshold=0.7, class_balancer=class_balancer_t if args.bct else None,
                            class_num=class_num, ignore_label=ignore_label)
    else:
        logger.info('>>>>>>> using CrossEntropyLoss.')
        loss_fn_t = CrossEntropy(ignore_label=ignore_label, class_balancer=class_balancer_t if args.bct else None)

    # source loader
    sourceloader = DALoader(cfg.SOURCE_DATA_CONFIG, cfg.DATASETS)
    sourceloader_iter = Iterator(sourceloader)
    # pseudo loader (target)
    pseudo_loader = DALoader(cfg.PSEUDO_DATA_CONFIG, cfg.DATASETS)
    # target loader
    targetloader = DALoader(cfg.TARGET_DATA_CONFIG, cfg.DATASETS)
    targetloader_iter = Iterator(targetloader)
    logger.info(f'batch num: source={len(sourceloader)}, target={len(targetloader)}, pseudo={len(pseudo_loader)}')
    # print(len(targetloader))
    epochs = stop_steps / len(sourceloader)
    logger.info('epochs ~= %.3f' % epochs)

    mIoU_max, iter_max = 0, 0
    optimizer = optim.SGD(model.parameters(),
                          lr=cfg.LEARNING_RATE, momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)
    for i_iter in tqdm(range(stop_steps)):

        lr = adjust_learning_rate(optimizer, i_iter, cfg)

        # Generate pseudo label
        # if i_iter % cfg.GENE_EVERY == 0:
        if i_iter == 0:
            if args.gen:
                if i_iter != 0:
                    shutil.move(f'{save_pseudo_label_path}_color',
                                f'{save_pseudo_label_path}_ssl_color_{i_iter - cfg.GENE_EVERY}')
                logger.info('###### Start generate pseudo dataset in round {}! ######'.format(i_iter))
                gener_target_pseudo(cfg, model, pseudo_loader, save_pseudo_label_path,
                                    size=eval(cfg.DATASETS).SIZE, save_prob=True, slide=True, ignore_label=ignore_label)
            target_config = cfg.TARGET_DATA_CONFIG
            target_config['mask_dir'] = [save_pseudo_label_path]
            logger.info(target_config)
            targetloader = DALoader(target_config, cfg.DATASETS)
            targetloader_iter = Iterator(targetloader)
            logger.info('###### Start model retraining dataset in round {}! ######'.format(i_iter))
        torch.cuda.synchronize()

        model.train()
        # source output
        batch_s = sourceloader_iter.next()
        images_s, label_s = batch_s[0]
        images_s, label_s = images_s.cuda(), label_s['cls'].cuda()
        # target output
        batch_t = targetloader_iter.next()
        images_t, label_t = batch_t[0]
        images_t, label_t_soft, regs_t = images_t.cuda(), label_t['cls'].cuda(), label_t['sup'].cuda()

        # model forward
        # source
        pred_s1, pred_s2, feat_s = model(images_s)
        # target
        pred_t1, pred_t2, feat_t = model(images_t)

        label_t_soft = aligner.label_refine(None, feat_t, [pred_t1, pred_t2], label_t_soft,
                                            refine=args.refine_label,
                                            mode=args.refine_mode,
                                            temp=args.refine_temp)
        label_t_hard = pseudo_selection(label_t_soft, cutoff_top=cfg.CUTOFF_TOP, cutoff_low=cfg.CUTOFF_LOW,
                                        return_type='tensor', ignore_label=ignore_label)

        # LRH
        if args.sam_refine:
            label_t_hard = homogenizer(label_t_hard, regs_t.squeeze(dim=1))

        aligner.update_prototype(feat_s, label_s)

        # loss
        loss_source = loss_calc([pred_s1, pred_s2], label_s, loss_fn=loss_fn_s, multi=True)
        if isinstance(loss_fn_t, UVEMLoss) or isinstance(loss_fn_t, UPSLoss):
            loss_target = loss_calc_uvem([pred_t1, pred_t2], label_t_hard, label_t_soft,
                                         loss_fn=loss_fn_t, multi=True)
        else:
            loss_target = loss_calc([pred_t1, pred_t2], label_t_hard, loss_fn=loss_fn_t, multi=True)

        loss = loss_source + loss_target

        optimizer.zero_grad()
        loss.backward()
        clip_grad.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()),
                                  max_norm=32, norm_type=2)
        optimizer.step()
        log_loss = f'iter={i_iter + 1}, total={loss:.3f}, loss_source={loss_source:.3f}, ' \
                   f'loss_target={loss_target:.3f},, lr = {lr:.3e}'

        # logging training process, evaluating and saving
        if i_iter == 0 or (i_iter + 1) % 50 == 0:
            logger.info(log_loss)
            if args.bcs:
                logger.info(str(loss_fn_s.class_balancer))
            if args.bct and i_iter >= cfg.FIRST_STAGE_STEP:
                logger.info(str(loss_fn_t.class_balancer))

        if i_iter == 0 or (i_iter + 1) % cfg.EVAL_EVERY == 0 or (i_iter + 1) >= stop_steps:
            ckpt_path = osp.join(cfg.SNAPSHOT_DIR, cfg.TARGET_SET + '_curr.pth')
            torch.save(model.state_dict(), ckpt_path)
            _, mIoU_curr = evaluate(model, cfg, True, ckpt_path, logger)
            if mIoU_max <= mIoU_curr:
                mIoU_max = mIoU_curr
                iter_max = i_iter + 1
                torch.save(model.state_dict(), osp.join(cfg.SNAPSHOT_DIR, cfg.TARGET_SET + '_best.pth'))
                if osp.isdir(os.path.join(cfg.SNAPSHOT_DIR, f'vis-{cfg.TARGET_SET}_best')):
                    shutil.rmtree(os.path.join(cfg.SNAPSHOT_DIR, f'vis-{cfg.TARGET_SET}_best'))
                shutil.copytree(os.path.join(cfg.SNAPSHOT_DIR, f'vis-{os.path.basename(ckpt_path)}'),
                                os.path.join(cfg.SNAPSHOT_DIR, f'vis-{cfg.TARGET_SET}_best'))
            logger.info(f'Best model in iter={iter_max}, best_mIoU={mIoU_max}.')
            model.train()

    logger.info(f'>>>> Usning {float(time.time() - time_from) / 3600:.3f} hours.')
    shutil.rmtree(save_pseudo_label_path, ignore_errors=True)
    logger.info('removing pseudo labels')

if __name__ == '__main__':
    # seed_torch(int(time.time()) % 10000019)
    seed_torch(2333)
    main()
