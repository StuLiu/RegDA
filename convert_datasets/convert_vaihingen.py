import argparse
import glob
import math
import os
import os.path as osp
import tempfile
import zipfile
import cv2
import mmcv
import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert vaihingen dataset to mmsegmentation format')
    parser.add_argument('dataset_path', help='vaihingen folder path')
    parser.add_argument('--tmp_dir', './tmp', help='path of the temporary directory')
    parser.add_argument('-o', '--out_dir', help='output path')
    parser.add_argument(
        '--clip_size',
        type=int,
        help='clipped size of image after preparation',
        default=512)
    parser.add_argument(
        '--stride_size',
        type=int,
        help='stride of clipping original images',
        default=256)
    args = parser.parse_args()
    return args


def clip_big_image(image_path, clip_save_dir, to_label=False):
    # Original image of Vaihingen dataset is very large, thus pre-processing
    # of them is adopted. Given fixed clip size and stride size to generate
    # clipped image, the intersection　of width and height is determined.
    # For example, given one 5120 x 5120 original image, the clip size is
    # 512 and stride size is 256, thus it would generate 20x20 = 400 images
    # whose size are all 512x512. IR-Red-Green
    image = mmcv.imread(image_path)     # Green-Red-IR

    h, w, c = image.shape
    cs = args.clip_size
    ss = args.stride_size

    num_rows = math.ceil((h - cs) / ss) if math.ceil(
        (h - cs) / ss) * ss + cs >= h else math.ceil((h - cs) / ss) + 1
    num_cols = math.ceil((w - cs) / ss) if math.ceil(
        (w - cs) / ss) * ss + cs >= w else math.ceil((w - cs) / ss) + 1

    x, y = np.meshgrid(np.arange(num_cols + 1), np.arange(num_rows + 1))
    xmin = x * cs
    ymin = y * cs

    xmin = xmin.ravel()
    ymin = ymin.ravel()
    xmin_offset = np.where(xmin + cs > w, w - xmin - cs, np.zeros_like(xmin))
    ymin_offset = np.where(ymin + cs > h, h - ymin - cs, np.zeros_like(ymin))
    boxes = np.stack([
        xmin + xmin_offset, ymin + ymin_offset,
        np.minimum(xmin + cs, w),
        np.minimum(ymin + cs, h)
    ],
        axis=1)

    if to_label:
        """
        num_classes: 6, ignore index is 255.

        CLASSES = ('ignore', 'impervious_surface', 'building', 'low_vegetation', 'tree',
                   'car', 'clutter')

        PALETTE = [[0, 0, 0], [255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0],
                   [255, 255, 0], [255, 0, 0]]

        """
        color_map = np.array([[0, 0, 0], [255, 255, 255], [255, 0, 0], [255, 255, 0],
                              [0, 255, 0], [0, 255, 255], [0, 0, 255]])     # B-G-R
        flatten_v = np.matmul(
            image.reshape(-1, c),
            np.array([2, 3, 4]).reshape(3, 1))
        out = np.ones_like(flatten_v) * 5
        for idx, class_color in enumerate(color_map):
            value_idx = np.matmul(class_color,
                                  np.array([2, 3, 4]).reshape(3, 1))
            out[flatten_v == value_idx] = idx
        image = out.reshape(h, w)
        image[image == 6] = 0   # merge background and clutter

    for box in boxes:
        start_x, start_y, end_x, end_y = box
        clipped_image = image[start_y:end_y, start_x:end_x] if to_label else image[start_y: end_y, start_x: end_x, :]
        area_idx = osp.basename(image_path).split('_')[3].strip('.tif')
        mmcv.imwrite(
            clipped_image.astype(np.uint8),
            osp.join(clip_save_dir,
                     f'{area_idx}_{start_x}_{start_y}_{end_x}_{end_y}.png'))


def main():
    splits = {
        'train': [
            'area1', 'area13', 'area17', 'area21',
            'area23', 'area26', 'area3', 'area32',
            'area37', 'area5', 'area7'
        ],
        'val': ['area11', 'area15', 'area28', 'area30', 'area34'],
        'test': [
            'area6', 'area24', 'area35', 'area16', 'area14', 'area22',
            'area10', 'area4', 'area2', 'area20', 'area8', 'area31', 'area33',
            'area27', 'area38', 'area12', 'area29'
        ],
    }

    dataset_path = args.dataset_path
    if args.out_dir is None:
        out_dir = osp.join('data', 'vaihingen2')
    else:
        out_dir = args.out_dir

    print('Making directories...')

    mmcv.mkdir_or_exist(osp.join(out_dir, 'img_dir', 'train'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'img_dir', 'val'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'img_dir', 'test'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'train'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'val'))
    mmcv.mkdir_or_exist(osp.join(out_dir, 'ann_dir', 'test'))

    zipp_list = glob.glob(os.path.join(dataset_path, '*.zip'))
    print('Find the data', zipp_list)

    with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmp_dir:
        for zipp in zipp_list:
            zip_file = zipfile.ZipFile(zipp)
            zip_file.extractall(tmp_dir)
            src_path_list = glob.glob(os.path.join(tmp_dir, '*.tif'))
            if zipp.endswith('ISPRS_semantic_labeling_Vaihingen.zip'):  # in zipp:
                src_path_list = glob.glob(
                    os.path.join(os.path.join(tmp_dir, 'top'), '*.tif'))
            if zipp.endswith('ISPRS_semantic_labeling_Vaihingen_ground_truth_COMPLETE.zip'):  # noqa
                src_path_list = glob.glob(os.path.join(tmp_dir, '*.tif'))
                # delete unused area9 ground truth
                for area_ann in src_path_list:
                    if 'area9' in area_ann:
                        src_path_list.remove(area_ann)
            prog_bar = mmcv.ProgressBar(len(src_path_list))
            for i, src_path in enumerate(src_path_list):
                area_idx = osp.basename(src_path).split('_')[3].strip('.tif')
                # data_type = 'train' if area_idx in splits['train'] else 'val'
                if area_idx in splits['train']:
                    data_type = 'train'
                elif area_idx in splits['val']:
                    data_type = 'val'
                else:
                    data_type = 'test'

                if 'top/' not in src_path:
                    dst_dir = osp.join(out_dir, 'ann_dir', data_type)
                    clip_big_image(src_path, dst_dir, to_label=True)
                else:
                    dst_dir = osp.join(out_dir, 'img_dir', data_type)
                    clip_big_image(src_path, dst_dir, to_label=False)
                # if 'noBoundary' in src_path:
                #     dst_dir = osp.join(out_dir, 'ann_dir', data_type)
                #     clip_big_image(src_path, dst_dir, to_label=True)
                # else:
                #     dst_dir = osp.join(out_dir, 'img_dir', data_type)
                #     clip_big_image(src_path, dst_dir, to_label=False)
                prog_bar.update()

        print('Removing the temporary files...')

    print('Done!')


def lbl2BMask():
    image = mmcv.imread('111.tif')
    h, w, c = image.shape
    color_map = np.array([[0, 0, 0], [255, 255, 255], [255, 0, 0],
                          [255, 255, 0], [0, 255, 0], [0, 255, 255], [0, 0, 255]])

    flatten_v = np.matmul(
        image.reshape(-1, c),
        np.array([2, 3, 4]).reshape(3, 1))
    out = np.ones_like(flatten_v) * 5
    for idx, class_color in enumerate(color_map):
        value_idx = np.matmul(class_color,
                              np.array([2, 3, 4]).reshape(3, 1))
        out[flatten_v == value_idx] = idx
    image = out.reshape(h, w, 1)

    image = np.where(image == 4, 255, 0)
    cv2.imwrite('ooo.png', image)


if __name__ == '__main__':
    args = parse_args()
    main()
    # lbl2BMask()
