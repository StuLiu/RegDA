"""
@Project : rsda
@File    : contrastive.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2023/3/31 下午8:53
@e-mail  : 1183862787@qq.com
"""
import torch
import torch.nn as nn
import torch.nn.functional as tnf


# "contrast": {
#       "proj_dim": 256,
#       "temperature": 0.1,
#       "base_temperature": 0.07,
#       "max_samples": 1024,
#       "max_views": 100,
#       "stride": 8,
#       "warmup_iters": 5000,
#       "loss_weight": 0.1,
#       "use_rmi": false
#     }


class PixelContrastLoss(nn.Module):

    def __init__(self, ):
        super(PixelContrastLoss, self).__init__()
        self.temperature = 0.1
        self.base_temperature = 0.07
        self.ignore_label = -1
        self.max_samples = 1024
        self.max_views = 100
        self.eps = 1e-5

    def _hard_anchor_sampling(self, feats, y_hat, y):
        """
        Args:
            feats: deep features. [b, h*w, k]
            y_hat: predicted values. [b, h*w]
            y: label. [b, h*w]
        Returns:

        """
        batch_size, feat_dim = feats.shape[0], feats.shape[-1]

        classes = []
        total_classes = 0
        for ii in range(batch_size):
            this_y = y_hat[ii]      # (h*w,)
            this_classes = torch.unique(this_y)     # the unique predicted labels
            this_classes = [x for x in this_classes if x != self.ignore_label]      # remove ignore_label
            this_classes = [x for x in this_classes if (this_y == x).nonzero().shape[0] > self.max_views]

            classes.append(this_classes)
            total_classes += len(this_classes)

        if total_classes == 0:
            return None, None

        n_view = self.max_samples // total_classes
        n_view = min(n_view, self.max_views)

        X_ = torch.zeros((total_classes, n_view, feat_dim), dtype=torch.float).cuda()
        y_ = torch.zeros(total_classes, dtype=torch.float).cuda()

        X_ptr = 0
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_y = y[ii]
            this_classes = classes[ii]

            for cls_id in this_classes:
                hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
                easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()

                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]

                if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                    num_hard_keep = n_view // 2
                    num_easy_keep = n_view - num_hard_keep
                elif num_hard >= n_view / 2:
                    num_easy_keep = num_easy
                    num_hard_keep = n_view - num_easy_keep
                elif num_easy >= n_view / 2:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep
                else:
                    Log.info('this shoud be never touched! {} {} {}'.format(num_hard, num_easy, n_view))
                    raise Exception

                perm = torch.randperm(num_hard)
                hard_indices = hard_indices[perm[:num_hard_keep]]
                perm = torch.randperm(num_easy)
                easy_indices = easy_indices[perm[:num_easy_keep]]
                indices = torch.cat((hard_indices, easy_indices), dim=0)

                X_[X_ptr, :, :] = feats[ii, indices, :].squeeze(1)
                y_[X_ptr] = cls_id
                X_ptr += 1

        return X_, y_

    def _contrastive(self, feats_, labels_):
        anchor_num, n_view = feats_.shape[0], feats_.shape[1]

        labels_ = labels_.contiguous().view(-1, 1)
        mask = torch.eq(labels_, torch.transpose(labels_, 0, 1)).float().cuda()

        contrast_count = n_view
        contrast_feature = torch.cat(torch.unbind(feats_, dim=1), dim=0)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, torch.transpose(contrast_feature, 0, 1)),
                                        self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        neg_mask = 1 - mask

        logits_mask = torch.ones_like(mask).scatter_(
            1, torch.arange(anchor_num * anchor_count).view(-1, 1).cuda(), 0)
        mask = mask * logits_mask

        neg_logits = torch.exp(logits) * neg_mask
        neg_logits = neg_logits.sum(1, keepdim=True)

        exp_logits = torch.exp(logits)

        log_prob = logits - torch.log(exp_logits + neg_logits + self.eps)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + self.eps)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

    def forward(self, feats, labels=None, predict=None):
        labels = labels.unsqueeze(1).float().clone()
        labels = torch.nn.functional.interpolate(labels,
                                                 (feats.shape[2], feats.shape[3]), mode='nearest')
        labels = labels.squeeze(1).long()       # (b, h, w)
        assert labels.shape[-1] == feats.shape[-1], '{} {}'.format(labels.shape, feats.shape)

        batch_size = feats.shape[0]

        labels = labels.contiguous().view(batch_size, -1)       # (b, h*w)
        predict = predict.contiguous().view(batch_size, -1)     # (b, h*w)
        feats = feats.permute(0, 2, 3, 1)       # (b, h, w, k)
        feats = feats.contiguous().view(feats.shape[0], -1, feats.shape[-1])    # (b, h*w, k)

        feats_, labels_ = self._hard_anchor_sampling(feats, labels, predict)

        loss = self._contrastive(feats_, labels_)
        return loss


if __name__ == '__main__':
    from regda.models.Encoder import Deeplabv2
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

    loss_instance = PixelContrastLoss()

    def rand_x_l():
        return torch.randn([8, 3, 512, 512]).float().cuda(), \
               torch.ones([8, 3, 512, 512]).float().cuda() / 2, \
               torch.randint(0, 1, [8, 512, 512]).long().cuda(), \
               torch.randint(1, 2, [8, 512, 512]).long().cuda()

    for i in range(5):

        optimizer.zero_grad()
        x_s, x_t, l_s, l_t = rand_x_l()
        _, p_s2, f_s = model(x_s)
        p_s2 = torch.argmax(p_s2, dim=1)
        # p_t1, p_t2, f_t = model(x_t)
        _loss = loss_instance(f_s, l_s, p_s2)
        print(_loss.cpu().item(), '\t', i)
        _loss.backward()
        for name, param in model.named_parameters():
            print(name, param.requires_grad)
            print(param.grad[0][0][0])
            break
    print('grad of loss instance')
    print('=========================================================')
