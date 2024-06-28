import shutil
import warnings
import os

from argparse import ArgumentParser
from skimage.io import imsave, imread

from regda.datasets import *
from regda.utils.tools import *
from regda.viz import VisualizeSegmm
from regda.models.Encoder import Deeplabv2


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    parser = ArgumentParser(description='Run predict methods.')
    parser.add_argument('config_path', type=str, help='config path')
    parser.add_argument('ckpt_path', type=str, help='ckpt path')
    parser.add_argument('image_path', type=str, help='ckpt path')
    parser.add_argument('--save-dir', type=str, default='./demo', help='save dir')
    parser.add_argument('--ins-norm', type=str2bool, default=True, help='save dir path')
    parser.add_argument('--slide', type=str2bool, default=True, help='save dir path')
    parser.add_argument('--tta', type=str2bool, default=False, help='save dir path')
    parser.add_argument('--gt', type=str2bool, default=False, help='save dir path')
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    cfg = import_config(args.config_path, copy=False, create=False)
    class_num = len(eval(cfg.DATASETS).LABEL_MAP)

    model_name = str(cfg.MODEL).lower()
    if model_name == 'resnet':
        model_name = 'resnet50'

    model = Deeplabv2(dict(
        backbone=dict(
            resnet_type=model_name,
            output_stride=16,
            pretrained=False,
        ),
        multi_layer=True,
        cascade=False,
        use_ppm=True,
        ppm=dict(
            num_classes=class_num,
            use_aux=False,
        ),
        inchannels=2048,
        num_classes=class_num,
        is_ins_norm=args.ins_norm,
    )).cuda()
    model_state_dict = torch.load(args.ckpt_path)
    model.load_state_dict(model_state_dict, strict=True)
    model.eval()
    viz_op = VisualizeSegmm(args.save_dir, eval(cfg.DATASETS).PALETTE)
    trans = cfg.TEST_DATA_CONFIG['transforms']
    with torch.no_grad():
        img = imread(args.image_path)
        img = trans(image=img)['image'].unsqueeze(dim=0).cuda()
        cls = pre_slide(model, img, num_classes=class_num, tta=args.tta) if args.slide else model(img)
        cls = cls.argmax(dim=1).cpu().numpy().squeeze()
        imsave(os.path.join(args.save_dir, 'prediction.png'), cls.astype(np.uint8))
        viz_op(cls, 'prediction_color.png')
        if args.gt:
            gt_path = str(args.image_path).replace('img_dir', 'ann_dir')
            if os.path.exists(gt_path):
                print(gt_path)
                viz_op(imread(gt_path), f'gt.png')

    torch.cuda.empty_cache()

