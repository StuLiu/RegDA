import torch.nn as nn
import torch.nn.functional as F
import torch
import ever as er
from regda.resnet import ResNetEncoder


class PPMBilinear(nn.Module):
    def __init__(self, num_classes=7, fc_dim=2048,
                 use_aux=False, pool_scales=(1, 2, 3, 6),
                 norm_layer=nn.BatchNorm2d
                 ):
        super(PPMBilinear, self).__init__()
        self.use_aux = use_aux
        self.ppm = []
        for scale in pool_scales:
            self.ppm.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(fc_dim, 512, kernel_size=1, bias=False),
                norm_layer(512),
                nn.ReLU(inplace=True)
            ))
        self.ppm = nn.ModuleList(self.ppm)
        if self.use_aux:
            self.cbr_deepsup = nn.Sequential(
                nn.Conv2d(fc_dim // 2, fc_dim // 4, kernel_size=3, stride=1,
                          padding=1, bias=False),
                norm_layer(fc_dim // 4),
                nn.ReLU(inplace=True),
            )
            self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_classes, 1, 1, 0)
            self.dropout_deepsup = nn.Dropout2d(0.1)

        self.conv_last = nn.Sequential(
            nn.Conv2d(fc_dim + len(pool_scales) * 512, 512,
                      kernel_size=3, padding=1, bias=False),
            norm_layer(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )

    def forward(self, conv_out):
        # conv5 = conv_out[-1]
        input_size = conv_out.size()
        ppm_out = [conv_out]
        for pool_scale in self.ppm:
            ppm_out.append(F.interpolate(
                pool_scale(conv_out),
                (input_size[2], input_size[3]),
                mode='bilinear', align_corners=False))

        ppm_out = torch.cat(ppm_out, 1)

        x = self.conv_last(ppm_out)

        if self.use_aux and self.training:
            conv4 = conv_out[-2]
            _ = self.cbr_deepsup(conv4)
            _ = self.dropout_deepsup(_)
            _ = self.conv_last_deepsup(_)

            return x
        else:
            return x


class Classifier_Module(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation,
                          bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out


class Deeplabv2(er.ERModule):
    def __init__(self, config):
        super(Deeplabv2, self).__init__(config)
        self.encoder = ResNetEncoder(self.config.backbone)
        if self.config.multi_layer:
            print('Use multi_layer!')
            if self.config.cascade:
                if self.config.use_ppm:
                    self.layer5 = PPMBilinear(**self.config.ppm1)
                    self.layer6 = PPMBilinear(**self.config.ppm2)
                else:
                    self.layer5 = self._make_pred_layer(Classifier_Module, self.config.inchannels // 2,
                                                        [6, 12, 18, 24], [6, 12, 18, 24], self.config.num_classes)
                    self.layer6 = self._make_pred_layer(Classifier_Module, self.config.inchannels,
                                                        [6, 12, 18, 24], [6, 12, 18, 24], self.config.num_classes)
            else:
                if self.config.use_ppm:
                    self.layer5 = PPMBilinear(**self.config.ppm)
                    self.layer6 = PPMBilinear(**self.config.ppm)
                else:
                    self.layer5 = self._make_pred_layer(Classifier_Module, self.config.inchannels,
                                                        [6, 12, 18, 24], [6, 12, 18, 24], self.config.num_classes)
                    self.layer6 = self._make_pred_layer(Classifier_Module, self.config.inchannels,
                                                        [6, 12, 18, 24], [6, 12, 18, 24], self.config.num_classes)
        else:
            if self.config.use_ppm:
                self.cls_pred = PPMBilinear(**self.config.ppm)
            else:
                self.cls_pred = self._make_pred_layer(Classifier_Module, self.config.inchannels,
                                                      [6, 12, 18, 24], [6, 12, 18, 24], self.config.num_classes)

        if self.config.is_ins_norm:
            if self.config.cascade:
                self.instance_norm1 = nn.InstanceNorm2d(self.config.inchannels)
                self.instance_norm2 = nn.InstanceNorm2d(self.config.inchannels)
            else:
                self.instance_norm = nn.InstanceNorm2d(self.config.inchannels)

    @staticmethod
    def _make_pred_layer(block, inplanes, dilation_series, padding_series, num_classes):
        return block(inplanes, dilation_series, padding_series, num_classes)

    def forward(self, x):
        if self.config.multi_layer:
            if self.config.cascade:
                feat1, feat2 = self.encoder(x)[-2:]
                if self.config.is_ins_norm:
                    feat1 = self.instance_norm1(feat1)
                    feat2 = self.instance_norm2(feat2)
                x1 = self.layer5(feat1)
                x2 = self.layer6(feat2)
                if self.training:
                    return x1, feat1, x2, feat2
                else:
                    x1 = F.interpolate(x1, x.shape[-2:], mode='bilinear', align_corners=True)
                    x2 = F.interpolate(x2, x.shape[-2:], mode='bilinear', align_corners=True)
                    return (x1.softmax(dim=1) + x2.softmax(dim=1)) / 2
            else:
                feat = self.encoder(x)[-1]
                if self.config.is_ins_norm:
                    feat = self.instance_norm(feat)
                x1 = self.layer5(feat)
                x2 = self.layer6(feat)
                if self.training:
                    return x1, x2, feat
                else:
                    x1 = F.interpolate(x1, x.shape[-2:], mode='bilinear', align_corners=True)
                    x2 = F.interpolate(x2, x.shape[-2:], mode='bilinear', align_corners=True)
                    return (x1.softmax(dim=1) + x2.softmax(dim=1)) / 2
        else:
            feat = self.encoder(x)[-1]
            if self.config.is_ins_norm:
                feat = self.instance_norm(feat)
            x1 = self.cls_pred(feat)
            if self.training:
                return x1, feat
            else:
                x1 = F.interpolate(x1, x.shape[-2:], mode='bilinear', align_corners=True)
                return x1.softmax(dim=1)

    def set_default_config(self):
        self.config.update(dict(
            backbone=dict(
                resnet_type='resnet50',
                output_stride=16,
                pretrained=True,
            ),
            multi_layer=False,
            cascade=False,
            use_ppm=False,
            ppm=dict(
                num_classes=7,
                use_aux=False,
                norm_layer=nn.BatchNorm2d,

            ),
            inchannels=2048,
            num_classes=7,
            is_ins_norm=False,
        ))


if __name__ == '__main__':
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
    x_i = torch.randn([8, 3, 512, 512]).cuda()
    rs = model(x_i)
    print(x_i.shape)
    pass
