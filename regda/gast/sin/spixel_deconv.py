import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, constant_
from regda.gast.sin.model_util import *
# from train_util import *

# define the function includes in import *
__all__ = [
    'SpixelNet1l','SpixelNet1l_bn'
]


class SpixelNet(nn.Module):
    expansion = 1

    def __init__(self, batchNorm=True):
        super(SpixelNet,self).__init__()

        self.batchNorm = batchNorm
        self.assign_ch = 9

        self.conv0a = conv(self.batchNorm, 3, 16, kernel_size=3, padding=1)
        self.conv0b = conv(self.batchNorm, 16, 16, kernel_size=3, padding=1)

        self.conv1a = conv(self.batchNorm, 16, 32, kernel_size=3, stride=2)
        self.conv1b = conv(self.batchNorm, 32, 32, kernel_size=3, padding=1)

        self.conv2a = conv(self.batchNorm, 32, 64, kernel_size=3, stride=2)
        self.conv2b = conv(self.batchNorm, 64, 64, kernel_size=3, padding=1)

        self.conv3a = conv(self.batchNorm, 64, 128, kernel_size=3, stride=2)
        self.conv3b = conv(self.batchNorm, 128, 128, kernel_size=3, padding=1)

        self.conv4a = conv(self.batchNorm, 128, 256, kernel_size=3, stride=2)
        self.conv4b = conv(self.batchNorm, 256, 256, kernel_size=3, padding=1)

        self.deconv3 = deconv(256, 128)
        self.deconv3_h = deconv_h(256, 128)
        self.deconv3_v = deconv_v(128, 128)
        self.conv3_1 = conv(self.batchNorm, 256, 128, padding=1)
        self.pred_mask3 = predict_mask(128, self.assign_ch)
        self.pred_mask3_h = predict_mask(128, 2)
        self.pred_mask3_v = predict_mask(128, 2)

        self.deconv2 = deconv(128, 64)
        self.deconv2_h = deconv_h(128, 64)
        self.deconv2_v = deconv_v(64, 64)
        self.conv2_1 = conv(self.batchNorm, 128, 64, padding=1)
        self.pred_mask2 = predict_mask(64, self.assign_ch)
        self.pred_mask2_h = predict_mask(64, 2)
        self.pred_mask2_v = predict_mask(64, 2)

        self.deconv1 = deconv(64, 32)
        self.deconv1_h = deconv_h(64, 32)
        self.deconv1_v = deconv_v(32, 32)
        self.conv1_1 = conv(self.batchNorm, 64, 32, padding=1)
        self.pred_mask1 = predict_mask(32, self.assign_ch)
        self.pred_mask1_h = predict_mask(32, 2)
        self.pred_mask1_v = predict_mask(32, 2)

        self.deconv0 = deconv(32, 16)
        self.deconv0_h = deconv_h(32, 16)
        self.deconv0_v = deconv_v(16, 16)
        self.conv0_1 = conv(self.batchNorm, 32 , 16, padding=1)
        self.pred_mask0 = predict_mask(16, self.assign_ch)
        self.pred_mask0_h = predict_mask(16, 2)
        self.pred_mask0_v = predict_mask(16, 2)

        self.softmax = nn.Softmax(1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x):
        out1 = self.conv0b(self.conv0a(x))  # 5*5
        out2 = self.conv1b(self.conv1a(out1))  # 11*11
        out3 = self.conv2b(self.conv2a(out2))  # 23*23
        out4 = self.conv3b(self.conv3a(out3))  # 47*47
        out5 = self.conv4b(self.conv4a(out4))  # 95*95

        out_deconv3_h = self.deconv3_h(out5)
        mask3_h = self.pred_mask3_h(out_deconv3_h)
        prob3_h = self.softmax(mask3_h)

        out_deconv3_v = self.deconv3_v(out_deconv3_h)
        mask3_v = self.pred_mask3_v(out_deconv3_v)
        prob3_v = self.softmax(mask3_v)

        out_deconv2_h = self.deconv2_h(out_deconv3_v)
        mask2_h = self.pred_mask2_h(out_deconv2_h)
        prob2_h = self.softmax(mask2_h)

        out_deconv2_v = self.deconv2_v(out_deconv2_h)
        mask2_v = self.pred_mask2_v(out_deconv2_v)
        prob2_v = self.softmax(mask2_v)

        out_deconv1_h = self.deconv1_h(out_deconv2_v)
        mask1_h = self.pred_mask1_h(out_deconv1_h)
        prob1_h = self.softmax(mask1_h)

        out_deconv1_v = self.deconv1_v(out_deconv1_h)
        mask1_v = self.pred_mask1_v(out_deconv1_v)
        prob1_v = self.softmax(mask1_v)

        out_deconv0_h = self.deconv0_h(out_deconv1_v)
        mask0_h = self.pred_mask0_h(out_deconv0_h)
        prob0_h = self.softmax(mask0_h)

        out_deconv0_v = self.deconv0_v(out_deconv0_h)
        mask0_v = self.pred_mask0_v(out_deconv0_v)
        prob0_v = self.softmax(mask0_v)

        return prob0_v, prob0_h, prob1_v, prob1_h, prob2_v, prob2_h, prob3_v, prob3_h

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


def SpixelNet1l( data=None):
    # Model without  batch normalization
    model = SpixelNet(batchNorm=False)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model


def SpixelNet1l_bn(data=None):
    # model with batch normalization
    model = SpixelNet(batchNorm=True)
    if data is not None:
        model.load_state_dict(data['state_dict'])
    return model
#
# SpixelNet1l()
# SpixelNet1l_bn()