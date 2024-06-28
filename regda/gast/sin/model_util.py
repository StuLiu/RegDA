import torch
import torch.nn as nn
import torch.nn.functional as F

# *************************** my functions ****************************


def predict_param(in_planes, channel=3):
    return nn.Conv2d(in_planes, channel, kernel_size=3, stride=1, padding=1, bias=True)


def predict_mask(in_planes, channel=9):
    return nn.Conv2d(in_planes, channel, kernel_size=3, stride=1, padding=1, bias=True)



def predict_feat(in_planes, channel=20, stride=1):
    return nn.Conv2d(in_planes, channel, kernel_size=3, stride=stride, padding=1, bias=True)


def predict_prob(in_planes, channel=9):
    return nn.Sequential(
        nn.Conv2d(in_planes, channel, kernel_size=3, stride=1, padding=1, bias=True),
        nn.Softmax(1)
    )

# ***********************************************************************


def conv(batchNorm, in_planes, out_planes, kernel_size=3, stride=1, padding=1):
    if batchNorm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.LeakyReLU(0.1)
        )


def deconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, bias=True),
        nn.LeakyReLU(0.1)
    )


def deconv_h(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=(1, 2), padding=(1, 1), bias=True),
        nn.LeakyReLU(0.1)
    )


def deconv_v(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=(2, 1), padding=(1, 1), bias=True),
        nn.LeakyReLU(0.1)
    )


def index_conv(x):
    # kernel_1 = torch.tensor([[[0, 0, 0], [0, 1, 0], [0, 0, 0]],\
    #                        [[0, 0, 0], [1, 0, 0], [0, 0, 0]],\
    #                        [[0, 0, 0], [0, 0, 0], [0, 1, 0]],\
    #                        [[0, 0, 0], [0, 0, 1], [0, 0, 0]],\
    #                        [[0, 1, 0], [0, 0, 0], [0, 0, 0]],\
    #                        [[0, 0, 0], [0, 0, 0], [1, 0, 0]],\
    #                        [[0, 0, 0], [0, 0, 0], [0, 0, 1]],\
    #                        [[0, 0, 1], [0, 0, 0], [0, 0, 0]],\
    #                        [[1, 0, 0], [0, 0, 0], [0, 0, 0]]]
    #                       )
    kernel = torch.tensor([[[1, 0, 0], [0, 0, 0], [0, 0, 0]],\
                           [[0, 0, 1], [0, 0, 0], [0, 0, 0]],\
                           [[0, 0, 0], [0, 0, 0], [0, 0, 1]],\
                           [[0, 0, 0], [0, 0, 0], [1, 0, 0]],\
                           [[0, 1, 0], [0, 0, 0], [0, 0, 0]],\
                           [[0, 0, 0], [0, 0, 1], [0, 0, 0]],\
                           [[0, 0, 0], [0, 0, 0], [0, 1, 0]],\
                           [[0, 0, 0], [1, 0, 0], [0, 0, 0]],\
                           [[0, 0, 0], [0, 1, 0], [0, 0, 0]]])

    # kernel = torch.tensor([[[0, 0, 0], [0, 0, 0], [0, 0, 0]],\
    #                        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],\
    #                        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],\
    #                        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],\
    #                        [[0, 1, 0], [0, 0, 0], [0, 0, 0]],\
    #                        [[0, 0, 0], [0, 0, 1], [0, 0, 0]],\
    #                        [[0, 0, 0], [0, 0, 0], [0, 1, 0]],\
    #                        [[0, 0, 0], [1, 0, 0], [0, 0, 0]],\
    #                        [[0, 0, 0], [0, 1, 0], [0, 0, 0]]])

    kernel = kernel.float().to(x.device)
    bz, c, h, w = x.shape
    # for odd size
    output = F.conv_transpose2d(x.float(), kernel.unsqueeze(0), stride=2, padding=0, output_padding=0)
    return output


def initialize_map(x):
    bz, c, h, w = x.shape
    h = (h+15)//16
    w = (w+15)//16
    device = x.device
    start_id = 1
    end_id = start_id + h*w
    map = torch.arange(start_id, end_id).reshape(h, w).float()
    batch_map = map.repeat(bz, 1, 1, 1).to(device)
    return batch_map


def update_map(prob, map):
    # prob: bz*9*h*w
    # map: bz*1*h'*w'
    map_ = index_conv(map)
    bz, c, h, w = prob.shape
    device = prob.device
    # map_one = (map_ != 0)
    map_one = -F.relu(-map_ + 1) + 1
    prob = prob * map_one
    index_map = torch.arange(0, c).reshape(1, c, 1, 1).to(device)
    max_prob, max_id = prob.max(dim=1, keepdim=True)
    assignment = F.relu(prob - max_prob - (index_map - max_id) * (index_map - max_id) + 1)

    # temp = torch.arange(0, c)
    # temp = temp.repeat(bz, h, w, 1).permute(0, 3, 1, 2).to(device)
    # assignment = torch.where(temp == max_id, torch.ones(bz, c, h, w).to(device), torch.zeros(bz, c, h, w).to(device))
    new_map_ = assignment.float() * map_
    new_map = torch.sum(new_map_, dim=1, keepdim=True)
    return prob, new_map


def update_h_map(prob, map):
    b, c, h, w = map.shape
    device = prob.device
    lr_map = map
    lr_map = F.interpolate(lr_map, (h, 2*w), mode='nearest')
    # lr_map = F.pad(lr_map, (1, 1, 0, 0), mode='replicate')
    left_p = lr_map[:, :, :, :-1]
    right_p = lr_map[:, :, :, 1:]
    lr_map = torch.cat((left_p, right_p), dim=1)

    index_map = torch.arange(0, 2).reshape(1, 2, 1, 1).to(device)
    max_prob, max_id = prob.max(dim=1, keepdim=True)
    assignment = F.relu(prob - max_prob - (index_map - max_id) * (index_map - max_id) + 1)
    new_map_ = assignment.float() * lr_map
    new_map = torch.sum(new_map_, dim=1, keepdim=True)
    return new_map


def update_v_map(prob, map):
    b, c, h, w = map.shape
    device = prob.device
    tb_map = map
    tb_map = F.interpolate(tb_map, (2*h, w), mode='nearest')
    # tb_map = F.pad(tb_map, (0, 0, 1, 1), mode='replicate')
    top_p = tb_map[:, :, :-1, :]
    bott_p = tb_map[:, :, 1:, :]
    tb_map = torch.cat((top_p, bott_p), dim=1)

    index_map = torch.arange(0, 2).reshape(1, 2, 1, 1).to(device)
    max_prob, max_id = prob.max(dim=1, keepdim=True)
    assignment = F.relu(prob - max_prob - (index_map - max_id) * (index_map - max_id) + 1)
    new_map_ = assignment.float() * tb_map
    new_map = torch.sum(new_map_, dim=1, keepdim=True)
    return new_map


def update_spixel_map(img, prob0_v, prob0_h, prob1_v, prob1_h, prob2_v, prob2_h, prob3_v, prob3_h):
    initial_map = initialize_map(img)
    map3_h = update_h_map(prob3_h, initial_map)
    map3_v = update_v_map(prob3_v, map3_h)
    map2_h = update_h_map(prob2_h, map3_v)
    map2_v = update_v_map(prob2_v, map2_h)
    map1_h = update_h_map(prob1_h, map2_v)
    map1_v = update_v_map(prob1_v, map1_h)
    map0_h = update_h_map(prob0_h, map1_v)
    map0_v = update_v_map(prob0_v, map0_h)

    return map0_v



if __name__ == '__main__':
    map = torch.tensor([[1, 2], [3, 4]]).unsqueeze(0).unsqueeze(0)
    output = index_conv(map)
    print(output)

