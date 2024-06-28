import os

from regda.datasets.daLoader import DALoader
import logging
logger = logging.getLogger(__name__)
from regda.utils.tools import *
from ever.util.param_util import count_model_parameters
from regda.viz import VisualizeSegmm
from argparse import ArgumentParser
from regda.datasets import *
from regda.gast.metrics import PixelMetricIgnore
from regda.utils.eval import evaluate


if __name__ == '__main__':
    seed_torch(2333)

    parser = ArgumentParser(description='Run predict methods.')
    parser.add_argument('--config-path', type=str, default='st.gast.2urban', help='config path')
    parser.add_argument('--ckpt-path', type=str, default='log/GAST/2urban_c_57.67_10000_40.67/URBAN10000.pth',
                        help='ckpt path')
    parser.add_argument('--multi-layer', type=str2bool, default=True, help='save dir path')
    parser.add_argument('--ins-norm', type=str2bool, default=True, help='is instance norm in net end?')
    parser.add_argument('--test', type=str2bool, default=False, help='evaluate the test set?')
    parser.add_argument('--tta', type=str2bool, default=False, help='save dir path')
    args = parser.parse_args()
    from regda.models.Encoder import Deeplabv2
    cfg = import_config(args.config_path, copy=False, create=False)
    log_dir = os.path.dirname(args.ckpt_path)
    cfg.SNAPSHOT_DIR = log_dir
    logger = get_console_file_logger(name='Baseline', logdir=log_dir)

    class_num = len(eval(cfg.DATASETS).LABEL_MAP)
    model_name = str(cfg.MODEL).lower()
    if model_name == 'resnet':
        model_name = 'resnet50'
    logger.info(model_name)

    model = Deeplabv2(dict(
        backbone=dict(
            resnet_type=model_name,
            output_stride=16,
            pretrained=True,
        ),
        multi_layer=args.multi_layer,
        cascade=False,
        use_ppm=True,
        ppm=dict(
            num_classes=class_num,
            use_aux=False,
        ),
        inchannels=2048,
        num_classes=class_num,
        is_ins_norm=args.ins_norm
    )).cuda()
    evaluate(model, cfg, False, args.ckpt_path, logger, tta=args.tta, test=args.test)
