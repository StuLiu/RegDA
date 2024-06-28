"""
@Project : rsda
@File    : triple.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2023/4/4 下午2:33
@e-mail  : 1183862787@qq.com
"""
import torch
import torch.nn as nn


class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Imported from `<https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py>`_.

    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """

    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (num_classes).
        """
        n = inputs.shape[0]

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        return self.ranking_loss(dist_an, dist_ap, y)


if __name__ == '__main__':
    from regda.models.Encoder import Deeplabv2
    from regda.gast.alignment import DownscaleLabel
    import torch.optim as optim

    model = Deeplabv2(dict(
        backbone=dict(
            resnet_type='resnet50',
            output_stride=16,
            pretrained=True,
        ),
        multi_layer=True,
        cascade=False,
        use_ppm=True,
        ppm=dict(
            num_classes=7,
            use_aux=False,
        ),
        inchannels=2048,
        num_classes=7,
        is_ins_norm=True,
    )).cuda()
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    loss_triple = TripletLoss()
    down = DownscaleLabel()

    def rand_x_l():
        return torch.randn([8, 3, 512, 512]).float().cuda(), \
               torch.ones([8, 3, 512, 512]).float().cuda() / 2, \
               torch.randint(0, 1, [8, 512, 512]).long().cuda(), \
               torch.randint(1, 2, [8, 512, 512]).long().cuda()

    for i in range(5):

        optimizer.zero_grad()
        x_s, x_t, l_s, l_t = rand_x_l()
        _, p_s2, f_s = model(x_s)
        f_s = f_s.permute(0, 2, 3, 1).reshape(-1, 2048)

        l_s = down(l_s).flatten()
        _loss = loss_triple(f_s, l_s)
        print(_loss.cpu().item(), '\t', i)
        _loss.backward()
        for name, param in model.named_parameters():
            print(name, param.requires_grad)
            print(param.grad[0][0][0])
            break
    print('grad of loss instance')
    print('=========================================================')
