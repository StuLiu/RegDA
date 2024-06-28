from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from glob import glob
from tqdm import tqdm
import cv2


class VisualizeSegmm(object):
    def __init__(self, out_dir, palette):
        self.out_dir = out_dir
        self.palette = palette
        os.makedirs(self.out_dir, exist_ok=True)

    def __call__(self, y_pred, filename):
        """
        Args:
            y_pred: 2-D or 3-D array of shape [1 (optional), H, W]
            filename: str
        Returns:
        """
        y_pred = y_pred.astype(np.uint8)
        y_pred = y_pred.squeeze()
        color_y = Image.fromarray(y_pred)
        color_y.putpalette(self.palette)
        color_y.save(os.path.join(self.out_dir, filename))


def vis_dir(input_dir, palette, offset=0):
    out_dir = input_dir + '_color'
    viser = VisualizeSegmm(out_dir, palette)
    img_paths = glob(r'' + input_dir + '/*.png')
    img_paths.sort()
    for img_path in tqdm(img_paths):
        pred = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) + offset   # if offset=-1, 0-7 -> -1, 6
        n = np.sum(pred == -1)
        if n > 0:
            print(np.unique(pred), n / float(pred.size), img_path)
        viser(pred, os.path.basename(img_path))
        # break


if __name__ == '__main__':
    from regda.datasets.loveda import LoveDA
    from regda.datasets.isprsda import IsprsDA

    # vis_dir(input_dir='../data/LoveDA/Train/Rural/masks_png', palette=LoveDA.PALETTE, offset=-1)
    # vis_dir(input_dir='../data/LoveDA/Train/Urban/masks_png', palette=LoveDA.PALETTE, offset=-1)
    # vis_dir(input_dir='../data/LoveDA/Val/Rural/masks_png', palette=LoveDA.PALETTE, offset=-1)
    # vis_dir(input_dir='../data/LoveDA/Val/Urban/masks_png', palette=LoveDA.PALETTE, offset=-1)

    vis_dir(input_dir='../data/IsprsDA/Potsdam/ann_dir/train', palette=IsprsDA.PALETTE, offset=0)
    vis_dir(input_dir='../data/IsprsDA/Potsdam/ann_dir/val', palette=IsprsDA.PALETTE, offset=0)
    vis_dir(input_dir='../data/IsprsDA/Potsdam/ann_dir/test', palette=IsprsDA.PALETTE, offset=0)
    vis_dir(input_dir='../data/IsprsDA/Vaihingen/ann_dir/train', palette=IsprsDA.PALETTE, offset=0)
    vis_dir(input_dir='../data/IsprsDA/Vaihingen/ann_dir/val', palette=IsprsDA.PALETTE, offset=0)
    vis_dir(input_dir='../data/IsprsDA/Vaihingen/ann_dir/test', palette=IsprsDA.PALETTE, offset=0)


# Generate datasets
def generate_data():
    """Generate 3 Gaussians samples with the same covariance matrix"""
    n, dim = 512, 3
    np.random.seed(0)
    C = np.array([[1., 0.2, 0], [0.15, 1, 0.2], [0.1, 0.4, 10.0]])
    x = np.r_[
        np.dot(np.random.randn(n, dim), C),
        np.dot(np.random.randn(n, dim), C) + np.array([1, 2, 5]),
        np.dot(np.random.randn(n, dim), C) + np.array([-5, -2, 3]),
    ]
    y = np.hstack((
        np.ones(n) * 0,
        np.ones(n) * 1,
        np.ones(n) * 2,
    ))
    return x, y


def generate_data2():
    """Generate 3 Gaussians samples with the same covariance matrix"""
    n, dim = 512, 2048
    x1 = torch.randn((n, dim))
    x2 = torch.randn((n, dim)) *2 - 3
    x3 = torch.randn((n, dim))  + 3
    x = torch.cat([x1, x2, x3], dim=0).float()
    y1 = torch.ones((n,))
    y2 = torch.ones((n,)) + 1
    y3 = torch.ones((n,)) + 2
    y = torch.cat([y1, y2, y3], dim=0)
    return x.cuda(), y.cuda()


class PCA(torch.nn.Module):

    def __init__(self, n_components=2):
        super().__init__()
        self.n_components = n_components

    def forward(self, x):
        n = x.shape[0]
        x_mean = torch.mean(x, dim=0)
        x_centered = x - x_mean
        covariance_matrix = 1 / n * torch.matmul(x_centered.T, x_centered)
        eigenvalues, eigenvectors = torch.eig(covariance_matrix, eigenvectors=True)
        eigenvalues = torch.norm(eigenvalues, dim=1)
        idx = torch.argsort(-eigenvalues)
        eigenvectors = eigenvectors[:, idx]
        proj_mat = eigenvectors[:, 0: self.n_components]
        x = x - x_mean
        return x.matmul(proj_mat)


# if __name__ == "__main__":
#     _input, _y = generate_data2()
#     pca = PCA(n_components=2)
#     trans_x = pca(_input)
#     _y = _y.cpu().numpy()
#     plt.scatter(trans_x.T[0].cpu().numpy(), trans_x.T[1].cpu().numpy(), c=_y)
#     plt.show()
