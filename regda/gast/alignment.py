"""
@Project : rsda
@File    : alignment.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2023/3/13 下午9:43
@e-mail  : 1183862787@qq.com
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as tnf
from torch_scatter import scatter
from regda.gast.class_ware_whiten import ClassWareWhitening
from regda.gast.coral import CoralLoss
from regda.gast.pseudo_generation import pseudo_selection
# from module.gast.superpixels import SuperPixelsModel


# from audtorch.metrics.functional import pearsonr
# from module.gast.mmd import MMDLoss


class Aligner:

    def __init__(self, logger, feat_channels=64, class_num=7, ignore_label=-1, decay=0.999, topk=32, resume=None):

        # channel number of feature maps
        self.feat_channels = feat_channels

        # the number of class in datasets
        self.class_num = class_num

        # the id of the ignored label
        self.ignore_label = ignore_label

        # decay for ema
        self.decay = decay

        # logging.logger for logging
        self.logger = logger

        # small float to avoid dividing 0.
        self.eps = 1e-7

        # topk number
        self.topk = topk
        self.topk_importance = torch.ones([1, topk, 1], dtype=torch.float).cuda()
        for i in range(topk):
            self.topk_importance[0][i][0] = 1 - 1.0 * i / topk

        # self.eps_max = torch.FloatTensor([1e7]).cuda()[0]

        # prototypes for all classes. Note that, the prototypes is computed by source features only.
        if resume:
            self.prototypes = torch.load(resume, map_location='cpu').cuda()
            self.logger.info(f'finish init prototypes!')
            self.logger.info(f'prototypes({self.prototypes.shape})={self.prototypes}')
        else:
            self.prototypes = torch.zeros([class_num, feat_channels], requires_grad=False).cuda()

        # downscale for gt label with full size
        self.downscale_gt = DownscaleLabel(scale_factor=16, n_classes=class_num, ignore_label=ignore_label,
                                           min_ratio=0.75)

        # criterion for domain alignment
        self.coral = CoralLoss()
        # self.mmd = MMDLoss(kernel_type='linear')

        # criterion for feature whitening
        self.whitener = ClassWareWhitening(class_ids=range(class_num), groups=32)  # self.feat_channels // 8)

        # superpixel model
        # self.sp_model = SuperPixelsModel(model_type='LSC')

        self._data_sum = torch.zeros([self.class_num, self.feat_channels], requires_grad=False).cuda()
        self._data_cnt = torch.zeros([self.class_num, 1], requires_grad=False).cuda()

    def align_domain(self, feat_s, feat_t):
        assert feat_s.shape == feat_t.shape, 'tensor "feat_s" has the same shape as tensor "feat_t"'
        assert len(feat_s.shape) == 4, 'tensor "feat_s" and "feat_t" must have 4 dimensions'
        feat_s = feat_s.permute(0, 2, 3, 1).reshape([-1, self.feat_channels])
        feat_t = feat_t.permute(0, 2, 3, 1).reshape([-1, self.feat_channels])
        return self.coral(feat_s, feat_t)

    def update_prototype(self, feat, label):
        """Update global prototypes by source features and labels."""
        label = self.downscale_gt(label)  # (b, 16*h, 16*w) -> (b, 1, h, w)
        self._compute_local_prototypes(feat, label, update=True, decay=self.decay)
        return label

    def update_prototype_bytarget(self, feat_t, label_t_soft):
        """Update global prototypes by source features and labels.
        Args:
            feat_t: (n, k, h, w)
            label_t_soft: (n, c, h, w)
        """
        b, k, h, w = feat_t.shape
        c = label_t_soft.shape[1]
        feat_t_flatten = feat_t.permute(0, 2, 3, 1).reshape(-1, 1, k)
        # (b, c, 16*h, 16*w) -> (b, c, h, w)
        label_t_soft_down = tnf.interpolate(label_t_soft, size=(h, w), mode='bilinear', align_corners=True)
        label_t_soft_down = label_t_soft_down.permute(0, 2, 3, 1).reshape(-1, c, 1)
        local_prototype = torch.mean(feat_t_flatten * label_t_soft_down, dim=0)  # (c, k)
        self.prototypes = self._ema(self.prototypes, local_prototype, self.decay).detach()

    def update_avg(self, feat, label) -> None:
        """Update global prototypes by source features and labels."""
        b, k, h, w = feat.shape
        feats = feat.permute(0, 2, 3, 1).reshape(-1, k)     # (b*h*w, k)
        feats = feats.unsqueeze(dim=1)                      # (b*h*w, 1, k)

        labels = self.downscale_gt(label)                   # (b, 16*h, 16*w) -> (b, 1, h, w)
        labels = self._index2onehot(labels)                 # (b, 1, h, w) -> (b*h*w, c)
        labels = labels.unsqueeze(dim=-1)                   # -> (b*h*w, c, 1)
        n_instance = torch.sum(labels, dim=0)               # (c, 1)

        self._data_sum = self._data_sum + torch.sum(feats * labels, dim=0).detach()      # (c, k)
        self._data_cnt = self._data_cnt + n_instance

    def init_avg(self):
        self.prototypes = self._data_sum / (self._data_cnt + self.eps)

        self.logger.info(f'finish init prototypes!')
        self.logger.info(f'examples cnt({self._data_cnt.shape})={self._data_cnt}')
        self.logger.info(f'prototypes({self.prototypes.shape})={self.prototypes}')

    def align_class(self, feat_s, label_s, feat_t=None, label_t=None):
        """ Compute the loss for class level alignment.
            Besides, update the shared prototypes by the local prototypes of source and target domain.
        Args:
            feat_s:  features from source, shape as (b, k, h, w)
            label_s: labels from source  , shape as (b, 16*h, 16*w)
            feat_t:  features from source, shape as (b, k, h, w)
            label_t: pseudo label, Tensor, shape as (b, 16*h, 16*w)
        Returns:
            loss_class: the loss for class level alignment
        """
        assert len(feat_s.shape) == 4, 'tensor "feat_s" and "feat_t" must have 4 dimensions'
        assert len(label_s.shape) >= 3, 'tensor "label_s" and "label_t" must have 3 dimensions'
        label_s = self.downscale_gt(label_s)  # (b, 16*h, 16*w) -> (b, 1, h, w)
        half = feat_s.shape[0] // 2
        local_prototype_s1 = self._compute_local_prototypes(feat_s[:half], label_s[:half], update=False)
        local_prototype_s2 = self._compute_local_prototypes(feat_s[half:], label_s[half:], update=False)
        local_prototype_s = self._compute_local_prototypes(feat_s, label_s, update=False)  # update global prototypes
        loss_inter = self._class_align_loss(local_prototype_s1, local_prototype_s2)
        if feat_t is None or label_t is None:
            loss_class = loss_inter
        else:
            label_t = self.downscale_gt(label_t)
            local_prototype_t = self._compute_local_prototypes(feat_t, label_t, update=False)
            loss_intra = self._class_align_loss(local_prototype_s, local_prototype_t)
            loss_class = 0.5 * (loss_inter + loss_intra)
        return loss_class

    def align_instance(self, feat_s, label_s, feat_t=None, label_t=None):
        label_s = self.downscale_gt(label_s)  # (b, 1, h, w)
        loss_instance = self._instance_align_loss(feat_s, label_s)
        if feat_t is not None and label_t is not None:
            label_t = self.downscale_gt(label_t)
            loss_instance += self._instance_align_loss(feat_t, label_t)
            loss_instance *= 0.5
        return loss_instance

    def whiten_class_ware(self, feat_s, label_s, feat_t=None, label_t=None):
        loss_white = self.whitener(feat_s, self.downscale_gt(label_s))
        if feat_t is not None and label_t is not None:
            loss_white += self.whitener(feat_t, self.downscale_gt(label_t))
            loss_white *= 0.5
        return loss_white

    def show(self, save_path=None, display=True):
        pass

    def superpixel_expand(self, label_t_hard, label_t_sup):
        """
        Args:
            label_t_hard: Tensor, pseudo labels, (b, 16*h, 16*w)
            label_t_sup: Tensor, superpixel label, (b, 1, 16*h, 16*w)
        Returns:
            label_t_hard: Tensor, refined pseudo labels, (b, 16*h, 16*w)
        """
        b, h, w = label_t_hard.shape
        label_t_onehot = self._index2onehot(label_t_hard)       # (b, c, 16*h, 16*w)
        label_t_onehot = label_t_onehot.reshape(b, -1, self.class_num)  # (b, 16*hx16*w, c)
        label_t_sup_ = label_t_sup.reshape(b, -1, 1)                                        # (b, 16*hx16*w, 1)
        class_num_4sup = scatter(src=label_t_onehot, index=label_t_sup_, dim=1, reduce='sum')  # (b, n_sup, c)
        class_num_max, class_num_max_id = torch.max(class_num_4sup, dim=-1, keepdim=True)   # (b, n_sup, 1)
        class_num_max_id[class_num_max == 0] = -1
        label_t_hard_ = torch.gather(class_num_max_id, dim=1, index=label_t_sup_)           # (b, 16*hx16*w, 1)
        label_t_hard_ = label_t_hard_.reshape(b, h, w)
        return label_t_hard_

    def label_refine(self, label_t_sup, feat_t, preds_t, label_t_soft, refine=True, mode='all', temp=2.0):
        """Refine the pseudo label online by the distances between features and prototypes.
        Args:
            label_t_sup: Tensor, superpixel label, (b, 1, 16*h, 16*w)
            feat_t: Tensor, feature map, (b, k, h, w)
            preds_t: Tensor or [Tensor, ], predicted logits, (b, c, h, w) or [(b, c, h, w), ]
            label_t_soft: Tensor, pseudo labels, (b, c, 16*h, 16*w)
            refine: bool, if refine the pseudo labels or not.
            mode: refine schema ['all', 's', 'p', 'n', 'l']
            temp:
        Returns:
            label_t_hard: Tensor, refined pseudo labels, (b, h, w)
        """
        assert mode in ['all', 's', 'p', 'n', 'l']

        if refine:
            weight = 0
            b, k, h, w = feat_t.shape
            h_origin, w_origin = label_t_soft.shape[-2:]
            feat_t_flat = feat_t.permute(0, 2, 3, 1).reshape(-1, k)  # (b*h*w, k)

            if mode in ['all', 'p']:  # prototype view
                simi_matrix = 1.0 / self._pearson_dist(feat_t_flat, self.prototypes)  # (b*h*w, c) Pearson distance
                # simi_matrix = -self._euclide_dist(feat_t_flat, self.prototypes, p=2)   # (b*h*w, c) Euclidean distance
                simi_matrix = simi_matrix.view(b, h, w, -1).permute(0, 3, 1, 2)  # (b, c, h, w)
                simi_matrix = tnf.interpolate(simi_matrix, (h_origin, w_origin),
                                              mode='bilinear', align_corners=True)  # (b, c, 16*h, 16*w)
                prototypes_weight = self._softmax_T(simi_matrix, temp=1, dim=1).detach()  # (b, c, 16*h, 16*w)
                prototypes_weight = prototypes_weight / (torch.max(prototypes_weight, dim=1, keepdim=True)[0] + 1e-7)
                weight += prototypes_weight

            if mode in ['all', 'l']:  # prediction view
                if isinstance(preds_t, list):
                    assert len(preds_t) == 2
                    x1 = tnf.interpolate(preds_t[0], (h_origin, w_origin), mode='bilinear', align_corners=True)
                    x2 = tnf.interpolate(preds_t[1], (h_origin, w_origin), mode='bilinear', align_corners=True)
                    prediction_weight = (self._softmax_T(x1, temp=temp, dim=1) +
                                         self._softmax_T(x2, temp=temp, dim=1)).detach() * 0.5
                else:
                    x = tnf.interpolate(preds_t, (h_origin, w_origin), mode='bilinear', align_corners=True)
                    prediction_weight = self._softmax_T(x, temp=temp, dim=1).detach()
                prediction_weight = prediction_weight / (torch.max(prediction_weight, dim=1, keepdim=True)[0] + 1e-7)
                weight += prediction_weight

            if label_t_sup is not None and mode in ['all', 's']:  # superpixel view
                # compute mean prob for each superpixel
                label_t_sup_ = label_t_sup.reshape(b, -1, 1)                                        # (b, 16h*16w, 1)
                sup_cnt = label_t_sup_.max()
                ignored = ((label_t_sup_ == sup_cnt).reshape(b, h_origin, w_origin, 1)
                           .permute(0, 3, 1, 2)).repeat(1, self.class_num, 1, 1)                    # (b, c, 16h, 16w)
                label_t_soft_ = label_t_soft.permute(0, 2, 3, 1).reshape(b, -1, self.class_num)     # (b, 16h*16w, c)
                mean_prob = scatter(src=label_t_soft_, index=label_t_sup_, dim=1, reduce='max')    # (b, n_sup+1, c)
                # retrieval probabilities for each pixel
                label_t_sup_ = label_t_sup_.repeat(1, 1, self.class_num)                            # (b, 16h*16w, c)
                prob_pixel = torch.gather(mean_prob, dim=1, index=label_t_sup_)                     # (b, 16h*16w, c)
                prob_pixel = (prob_pixel.reshape(b, h_origin, w_origin, self.class_num)
                              .permute(0, 3, 1, 2))                                                 # (b, c, 16h, 16w)
                # get similarity for each pixel
                prob_pixel = self._softmax_T(prob_pixel, temp=temp, dim=1).detach()                 # (b, c, 16h, 16w)
                sup_weight = prob_pixel / (torch.max(prob_pixel, dim=1, keepdim=True)[0] + 1e-7)    # (b, c, 16h, 16w)
                if mode == 'all':
                    weight = torch.where(ignored, weight, weight * sup_weight)
                else:
                    weight = torch.ones_like(sup_weight)
                    weight = torch.where(ignored, weight, sup_weight)

            if isinstance(weight, int):
                return label_t_soft

            label_t_soft = weight.detach() * label_t_soft
            label_t_soft = self._logits_norm(label_t_soft)
        return label_t_soft

    def get_prototype_weight_4pixel(self, feats, label_hard, temp=2.0):
        b, k, h, w = feats.shape
        _, h2, w2 = label_hard.shape
        _feats = feats.permute(0, 2, 3, 1).reshape(-1, k)  # (b*h*w, k)
        simi_matrix = 1.0 / self._pearson_dist(_feats, self.prototypes)  # (b*h*w, c) Pearson distance
        simi_matrix = simi_matrix.view(b, h, w, -1).permute(0, 3, 1, 2)  # (b, c, h, w)
        simi_matrix = tnf.interpolate(simi_matrix, label_hard.shape[-2:],
                                      mode='bilinear', align_corners=True)  # (b, c, 16*h, 16*w)
        # simi_matrix = torch.softmax(simi_matrix, dim=1)      # (b, c, 16*h, 16*w)
        simi_matrix = self._softmax_T(simi_matrix, temp=1, dim=1)  # (b, c, 16*h, 16*w)
        max_v, _ = torch.max(simi_matrix, dim=1, keepdim=True)
        simi_matrix = simi_matrix / (max_v + self.eps)
        label_onehot = self._index2onehot(label_hard).reshape(b, h2, w2, -1).permute(0, 3, 1, 2)  # (b, c, 16*h, 16*w)
        weight = torch.sum(simi_matrix * label_onehot, dim=1).reshape(-1)
        return weight.detach()  # (b* 16*h* 16*w)

    @staticmethod
    def _softmax_T(feat, temp=1.0, dim=1):
        assert temp > 0
        return torch.softmax(feat / temp, dim=dim)

    def _logits_norm(self, logits):
        """ Keep the sum==1 in class channel.
        Args:
            logits: Tensor, probabilities, shape=(b, c, h, w)
        Returns:
            logits_normed: Tensor, normalized probabilities, shape=(b, c, h, w)
        """
        assert len(logits.shape) == 4
        logits_sum = torch.sum(logits, dim=1, keepdim=True)
        logits_normed = logits / (logits_sum + self.eps)
        return logits_normed

    def _compute_local_prototypes(self, feat, label, update=False, decay=0.999):
        """Compute prototypes within a mini-batch
        Args:
            feat: torch.Tensor, mini-batch features, shape=(b, k, h, w)
            label: torch.Tensor, label(the gt or pseudo label, instead of logits), shape=(b, 1, h, w)
            update: bool, if update the global prototypes
            decay: float in (0, 1), the parameter for ema algorithm. the higher, update slower.

        Returns:
            local_prototype: class prototypes within a mini-batch. shape=(c, k)
        """
        assert 0 < decay < 1
        b, k, h, w = feat.shape
        feats = feat.permute(0, 2, 3, 1).reshape(-1, k)  # (b*h*w, k)
        feats = feats.unsqueeze(dim=1)  # (b*h*w, 1, k)

        labels = self._index2onehot(label)  # (b, 1, h, w) -> (b*h*w, c)
        labels = labels.unsqueeze(dim=-1)  # (b*h*w, c, 1)

        n_instance = labels.sum(0).expand(self.class_num, k)  # (c, k)
        local_prototype = (feats * labels).sum(0) / (n_instance + self.eps)  # (c, k)
        # 某些类别无样本，可能会生成0向量，参与类别对齐将导致退化，降低模型性能
        local_prototype = torch.where(n_instance < 1, self.prototypes, local_prototype)

        if update:
            self.prototypes = self._ema(self.prototypes, local_prototype, decay).detach()
            # print(self.prototypes)
        return local_prototype

    def _class_align_loss(self, prototypes_1, prototypes_2, margin=0.3, hard_ratio=0.3):
        """ Compute the loss between two local prototypes.
        Args:
            prototypes_1: local prototype from a batch. shape=(c, k)
            prototypes_2: local prototype from a batch. shape=(c, k)
            margin: distance for margin loss. a no-neg float.
            hard_ratio: ratio for selecting hardest examples.
        Returns:
            loss_2p: loss for two local prototypes.
        """
        assert prototypes_1.shape == prototypes_2.shape
        # return tnf.mse_loss(prototypes_1, prototypes_2)

        # compute distance matrix for each class pair
        # dist_matrix = torch.cdist(prototypes_1, prototypes_2, p=2)  # (c, c), Euclidean distances
        dist_matrix = self._pearson_dist(prototypes_1, prototypes_2)  # (c, c), pearson distances

        # hard example mining
        hard_num = min(math.ceil(hard_ratio * self.class_num), self.class_num - 1)  # int
        eye_neg = 1 - torch.eye(self.class_num).cuda()  # 1 - I, (c, c)
        dist_hardest, _ = torch.topk(dist_matrix * eye_neg, k=hard_num + 1, dim=1, largest=False)  # (c, hard_num + 1)

        d_pos = torch.diag(dist_matrix).unsqueeze(dim=-1)  # (c, 1)
        d_neg = dist_hardest[:, 1:]  # (c, hard_num)
        # loss_p2p = d_mean_pos / (d_mean_neg + self.eps)
        loss_p2p = (d_pos - d_neg + margin).max(torch.Tensor([1e-6]).cuda()[0])

        return loss_p2p.mean()

    def _instance_align_loss(self, feat, label, margin=0.3, hard_ratio=0.3):
        """Compute the loss between instances and prototypes.
        Args:
            feat: deep features outputted by backbone. shape=(b, k, h, w)
            label: gt or pseudo label. shape=(b, 1, h, w)
            margin: distance for margin loss. a no-neg float.
            hard_ratio: ratio for selecting hardest examples.
        Returns:
            loss_2p: loss between instances and their prototypes.
        """
        # b, k, h, w = feat.shape
        feat = feat.permute(0, 2, 3, 1).reshape(-1, self.feat_channels)  # (b*h*w, k)
        ignored = (label < 0).permute(0, 2, 3, 1).reshape(-1, 1)  # (b*h*w, 1)
        no_ignored = 1 - ignored.float()  # (b*h*w, 1)
        mask_pos = self._index2onehot(label)  # (b*h*w, c)
        mask_neg = 1 - mask_pos  # (b*h*w, c)

        # compute the dist between instances and classes
        # dist_matrix = torch.cdist(feat, self.prototypes, p=2)  # Euclidean distance, (b*h*w, c)
        dist_matrix = self._pearson_dist(feat, self.prototypes)  # Euclidean distance, (b*h*w, c)

        # compute the distances for each instance with their nearest prototypes.
        hard_num = min(math.ceil(hard_ratio * self.class_num) + 1, self.class_num)

        dist_hardest, _ = torch.topk(dist_matrix * mask_neg, k=hard_num, dim=1, largest=False)

        # the mean distance between instances and their prototypes
        # d_mean_pos = (dist_matrix * mask_pos).sum(dim=1)
        # d_mean_neg = dist_hardest.sum(dim=1) / ((hard_num - 1) + self.eps)
        d_pos = torch.sum(dist_matrix * mask_pos, dim=1, keepdim=True)  # (b*h*w, 1)
        d_neg = dist_hardest[:, 1:]  # (b*h*w, hard_num - 1)
        # loss_i2p = d_mean_pos / (d_mean_neg + self.eps)
        loss_i2p = (d_pos - d_neg + margin).max(torch.Tensor([1e-6]).cuda()[0])  # (b*h*w, hard_num - 1)
        loss_i2p *= no_ignored
        cnt_i2p = torch.sum(no_ignored) * (hard_num - 1)

        return loss_i2p.sum() / (cnt_i2p + self.eps)

    def _pearson_dist(self, feat1, feat2):
        """
        Compute the pearson distance between the representation vector of each instance
        Args:
            feat1: torch.FloatTensor, (n, k)
            feat2: torch.FloatTensor, (m, k)

        Returns:
            pearson_dist: (n, m), from 0 to 1
        """
        assert feat1.shape[-1] == feat2.shape[-1]
        k = feat1.shape[-1]
        centered_feat1 = feat1 - feat1.mean(dim=-1, keepdim=True)  # (n, k)
        centered_feat2 = feat2 - feat2.mean(dim=-1, keepdim=True)  # (m, k)
        centered_feat1 = centered_feat1.unsqueeze(dim=1)
        centered_feat2 = centered_feat2.unsqueeze(dim=0)
        covariance = (centered_feat1 * centered_feat2).sum(dim=-1, keepdim=False)  # (n,  m)

        bessel_corrected_covariance = covariance / (k - 1 + self.eps)  # (n,  m)

        feat1_std = feat1.std(dim=-1, keepdim=False)  # (n,)
        feat2_std = feat2.std(dim=-1, keepdim=False)  # (m,)
        feat1_std = feat1_std.unsqueeze(dim=1)  # (n, 1)
        feat2_std = feat2_std.unsqueeze(dim=0)  # (1 ,m)
        div_mat = feat1_std * feat2_std  # (n, m)
        pearson_dist = (-1.0 * bessel_corrected_covariance / (div_mat + self.eps) + 1.0) * 0.5

        return pearson_dist.detach()  # (n, m)

    def _compute_similarity(self, feat_1, feat_2, step=8):
        assert len(feat_1.shape) == 2
        simi = torch.zeros((feat_1.shape[0], feat_2.shape[0])).cuda()
        ii = int(feat_2.shape[0]) // step
        for i in range(ii):
            # simi_local = 1 / (self.eps + self._pearson_dist(feat_1, feat_2[i*step: (i+1)*step,:]))
            simi[:, i * step: (i + 1) * step] = 1 / (
                        self.eps + self._pearson_dist(feat_1, feat_2[i * step: (i + 1) * step, :]))
        return simi

    @staticmethod
    def _ema(history, curr, decay=0.999):
        new_average = (1.0 - decay) * curr + decay * history
        return new_average

    def _index2onehot(self, label):
        """Compute the one-hot label
        Args:
            label: torch.Tensor, gt or pseudo label, shape=(b, 1, h, w)
        Returns:
            labels: (b*h*w, c)
        """
        labels = label.clone()
        if len(label.shape) < 4:
            labels = labels.unsqueeze(dim=1)
        labels = labels.permute(0, 2, 3, 1).reshape(-1, 1)  # (b*h*w, 1)
        labels[labels == self.ignore_label] = self.class_num
        labels = tnf.one_hot(labels.squeeze(1), num_classes=self.class_num + 1)[:, :-1]  # (b*h*w, c)
        return labels


class DownscaleLabel(nn.Module):

    def __init__(self, scale_factor=16, n_classes=7, ignore_label=-1, min_ratio=0.75):
        super().__init__()
        assert scale_factor > 1
        self.scale_factor = scale_factor
        self.n_classes = n_classes
        self.ignore_label = ignore_label
        self.min_ratio = min_ratio

    def forward(self, label):
        label = label.clone()
        if len(label.shape) == 4:
            label = label.squeeze(dim=1)
        assert len(label.shape) == 3
        bs, orig_h, orig_w = label.shape
        trg_h, trg_w = orig_h // self.scale_factor, orig_w // self.scale_factor
        label[label == self.ignore_label] = self.n_classes
        out = tnf.one_hot(label, num_classes=self.n_classes + 1).permute(0, 3, 1, 2)
        assert list(out.shape) == [bs, self.n_classes + 1, orig_h, orig_w], out.shape
        out = tnf.avg_pool2d(out.float(), kernel_size=self.scale_factor)
        max_ratio, out = torch.max(out, dim=1, keepdim=True)
        out[out == self.n_classes] = self.ignore_label
        out[max_ratio < self.min_ratio] = self.ignore_label
        assert list(out.shape) == [bs, 1, trg_h, trg_w], out.shape
        return out


if __name__ == '__main__':
    from regda.models.Encoder import Deeplabv2
    import torch.optim as optim
    import logging

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

    aligner = Aligner(logger=logging.getLogger(''), feat_channels=2048, class_num=7)


    def rand_x_l():
        return torch.randn([8, 3, 512, 512]).float().cuda(), \
            torch.ones([8, 3, 512, 512]).float().cuda() / 2, \
            torch.randint(-1, 1, [8, 512, 512]).long().cuda(), \
            torch.randint(1, 2, [8, 512, 512]).long().cuda()


    x_s, x_t, l_s, l_t = rand_x_l()
    _, _, f_s = model(x_s)

    w = aligner.get_prototype_weight_4pixel(f_s, l_s)
    print(w.shape)

    for i in range(2):
        x_s, x_t, l_s, l_t = rand_x_l()
        _, _, f_s = model(x_s)
        p_1, p_2, f_t = model(x_t)
        # l_t = aligner.label_refine(f_t, [p_1, p_2], l_t, False)
        _loss_white = aligner.whiten_class_ware(f_s, l_s, f_t, l_t)
        print(_loss_white.cpu().item(), '\t', i)
        optimizer.zero_grad()
        _loss_white.backward()
        for name, param in model.named_parameters():
            print(name, param.requires_grad)
            print(param.grad[0][0][0])
            break
    print('grad of loss white')
    print('=========================================================')

    for i in range(2):
        x_s, x_t, l_s, l_t = rand_x_l()
        _, _, f_s = model(x_s)
        _, _, f_t = model(x_t)
        _loss_domain = aligner.align_domain(f_s, f_t)
        print(_loss_domain)
        optimizer.zero_grad()
        _loss_domain.backward()
        for name, param in model.named_parameters():
            print(name, param.requires_grad)
            print(param.grad[0][0][0])
            break
    print('grad of loss domain')
    print('=========================================================')

    for i in range(2):
        x_s, x_t, l_s, l_t = rand_x_l()
        _, _, f_s = model(x_s)
        p_1, p_2, f_t = model(x_t)
        # l_t = aligner.label_refine(f_t, [p_1, p_2], l_t, False)
        _loss_class = aligner.align_class(f_s, l_s, f_t, l_t)
        print(_loss_class)
        optimizer.zero_grad()
        _loss_class.backward()
        for name, param in model.named_parameters():
            print(name, param.requires_grad)
            print(param.grad[0][0][0])
            break
    print('grad of loss class ')
    print('=========================================================')

    for i in range(2):
        x_s, x_t, l_s, l_t = rand_x_l()
        _, _, f_s = model(x_s)
        p_1, p_2, f_t = model(x_t)
        # l_t = aligner.label_refine(f_t, [p_1, p_2], l_t, False)
        _loss_ins = aligner.align_instance(f_s, l_s, f_t, l_t)
        print(_loss_ins)
        optimizer.zero_grad()
        _loss_ins.backward()
        for name, param in model.named_parameters():
            print(name, param.requires_grad)
            print(param.grad[0][0][0])
            break
    print('grad of loss instance')
    print('=========================================================')

    from regda.utils.tools import loss_calc

    for i in range(2):
        x_s, x_t, l_s, l_t = rand_x_l()
        os1, os2, f_s = model(x_s)
        ot1, ot2, f_t = model(x_t)
        _loss_seg = loss_calc([os1, os2], l_s, multi=True)
        _loss_seg += loss_calc([ot1, ot2], l_t, multi=True)
        print(_loss_seg)
        optimizer.zero_grad()
        _loss_seg.backward()
        for name, param in model.named_parameters():
            print(name, param.requires_grad)
            print(param.grad[0][0][0])
            break
    print('grad of loss seg')
    print('=========================================================')
    print('end')
