"""
@Project : rsda
@File    : MSCLoss.py
@IDE     : PyCharm
@Author  : Wang Liu
@Date    : 2023/3/31 下午7:41
@e-mail  : 1183862787@qq.com
"""
# https://github.com/astuti/ILA-DA/blob/main/criterion_factory.py
import torch.nn as nn
import torch


class MSCLoss(nn.Module):
    def __init__(self, config_data):
        super(MSCLoss, self).__init__()
        self.m = config_data['m']
        self.mu = config_data['mu']  # mu in number
        self.eps = 1e-9
        self.k = config_data['k']  # k for knn

    def __get_sim_matrix(self, out_src, out_tar):
        matrix = torch.cdist(out_src, out_tar)
        matrix = matrix + 1.0
        matrix = 1.0 / matrix
        return matrix

    def __target_pseudolabels(self, sim_matrix, src_labels):
        ind = torch.sort(sim_matrix, descending=True, dim=0).indices
        ind_split = torch.split(ind, 1, dim=1)
        ind_split = [id.squeeze() for id in ind_split]
        vr_src = src_labels.unsqueeze(-1).repeat(1, self.n_per_domain)
        label_list = []
        for i in range(0, self.n_per_domain):
            _row = ind_split[i].long()
            _col = (torch.ones(self.n_per_domain) * i).long()
            _val = vr_src[_row, _col]
            top_n_val = _val[[j for j in range(0, self.k)]]
            label_list.append(top_n_val)

        all_top_labels = torch.stack(label_list, dim=1)
        assigned_target_labels = torch.mode(all_top_labels, dim=0).values
        return assigned_target_labels

    def forward(self, src_features, src_labels, tgt_features):

        sim_matrix = self.__get_sim_matrix(src_features, tgt_features)
        flat_src_labels = src_labels.squeeze()

        assigned_tgt_labels = self.__target_pseudolabels(sim_matrix, src_labels)

        simratio_score = []  # sim-ratio (knn conf measure) for all tgt

        sim_matrix_split = torch.split(sim_matrix, 1, dim=1)
        sim_matrix_split = [_id.squeeze() for _id in sim_matrix_split]

        r = self.m
        for i in range(0, self.n_per_domain):  # nln: nearest like neighbours, nun: nearest unlike neighbours
            t_label = assigned_tgt_labels[i]
            nln_mask = (flat_src_labels == t_label)
            nln_sim_all = sim_matrix_split[i][nln_mask]
            nln_sim_r = torch.narrow(torch.sort(nln_sim_all, descending=True)[0], 0, 0, r)

            nun_mask = ~(flat_src_labels == t_label)
            nun_sim_all = sim_matrix_split[i][nun_mask]
            nun_sim_r = torch.narrow(torch.sort(nun_sim_all, descending=True)[0], 0, 0, r)

            conf_score = (1.0 * torch.sum(nln_sim_r) / torch.sum(nun_sim_r)).item()  # sim ratio : confidence score
            simratio_score.append(conf_score)

        sort_ranking_score, ind_tgt = torch.sort(torch.tensor(simratio_score), descending=True)
        top_n_tgt_ind = torch.narrow(ind_tgt, 0, 0, self.mu)  # take top mu confident tgt labels

        filtered_sim_matrix_list = []

        for idx in top_n_tgt_ind:
            filtered_sim_matrix_list.append(sim_matrix_split[idx])

        filtered_sim_matrix = torch.stack(filtered_sim_matrix_list, dim=1)
        filtered_tgt_labels = assigned_tgt_labels[top_n_tgt_ind]
        loss = self.calc_loss(filtered_sim_matrix, src_labels, filtered_tgt_labels)
        return loss

    def __build_mask(self, vr_src, hr_tgt, n_tgt, same=True):
        if same:
            mask = (vr_src == hr_tgt).float()
        else:
            mask = (~(vr_src == hr_tgt)).float()

        sum_row = (torch.sum(mask, dim=1)) > 0.

        to_remove = [torch.ones(n_tgt) if (_s) else torch.zeros(n_tgt) for _s in sum_row]
        to_remove = torch.stack(to_remove, dim=0)
        return to_remove

    def calc_loss(self, filtered_sim_matrix, src_labels, filtered_tgt_labels):
        n_src = src_labels.shape[0]
        n_tgt = filtered_tgt_labels.shape[0]

        vr_src = src_labels.unsqueeze(-1).repeat(1, n_tgt)
        hr_tgt = filtered_tgt_labels.unsqueeze(-2).repeat(n_src, 1)

        mask_sim = (vr_src == hr_tgt).float()

        same = self.__build_mask(vr_src, hr_tgt, n_tgt, same=True)
        diff = self.__build_mask(vr_src, hr_tgt, n_tgt, same=False)

        if torch.cuda.is_available():
            same, diff, mask_sim = same.cuda(), diff.cuda(), mask_sim.cuda()

        final_mask = (same.bool() & diff.bool())

        if torch.cuda.is_available():
            filtered_sim_matrix, final_mask = filtered_sim_matrix.cuda(), final_mask.cuda()

        filtered_sim_matrix[~final_mask] = float('-inf')
        unpurged_exp_scores = torch.softmax(filtered_sim_matrix, dim=1)

        exp_scores = unpurged_exp_scores[~torch.isnan(unpurged_exp_scores.sum(dim=1))]
        sim_labels = mask_sim[~torch.isnan(unpurged_exp_scores.sum(dim=1))]

        contrastive_matrix = torch.sum(exp_scores * sim_labels, dim=1)
        loss = -1 * torch.mean(torch.log(contrastive_matrix))
        return loss