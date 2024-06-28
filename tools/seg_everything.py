"""
@Project : gstda
@File    : local_region_homog.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2023/12/18 下午4:58
@e-mail  : 1183862787@qq.com
"""
import warnings
from regda.utils.local_region_homog import SAM, get_all_regs


warnings.filterwarnings('ignore')


if __name__ == '__main__':
    sam_model = SAM(sam_checkpoint="ckpts/sam_vit_b_01ec64.pth", model_type="vit_b")
    sam_model.get_local_regions(
        image_path='data/IsprsDA/Vaihingen/img_dir/train/area1_0_0_512_512.png',
        save=False,
        show=True
    )
    get_all_regs(img_dir_tgt='data/IsprsDA/Vaihingen/img_dir/train')
    get_all_regs(img_dir_tgt='data/IsprsDA/Potsdam/img_dir/train')
