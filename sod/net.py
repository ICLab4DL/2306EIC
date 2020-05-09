#!/usr/bin/python3
# coding=utf-8

import torch.nn.functional as F
from res2net import *


class HM(nn.Module):
    def __init__(self, in_channel_left, in_channel_down):
        super(HM, self).__init__()
        self.conv0 = nn.Conv2d(in_channel_left, 256, kernel_size=1, stride=1, padding=0)
        self.bn0 = nn.BatchNorm2d(256)
        self.conv1 = nn.Conv2d(in_channel_down, 256, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

    def forward(self, left, down):
        left = F.relu(self.bn0(self.conv0(left)), inplace=True)  # 256
        down = down.mean(dim=(2, 3), keepdim=True)
        down = F.relu(self.conv1(down), inplace=True)
        down = torch.sigmoid(self.conv2(down))
        return left * down

    def initialize(self):
        weight_init(self)


class SR(nn.Module):
    def __init__(self, in_channel):
        super(SR, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, 256, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out1 = F.relu(self.bn1(self.conv1(x)), inplace=True)  # 256
        out2 = self.conv2(out1)
        w, b = out2[:, :256, :, :], out2[:, 256:, :, :]

        return F.relu(w * out1 + b, inplace=True)

    def initialize(self):
        weight_init(self)


class Fusion(nn.Module):
    def __init__(self, in_channel_left, in_channel_down, in_channel_right):
        super(Fusion, self).__init__()
        # self.conv0 = nn.Conv2d(in_channel_left, 256, kernel_size=1, stride=1, padding=0)
        self.conv0 = nn.Conv2d(in_channel_left, 256, kernel_size=3, stride=1, padding=1)
        self.bn0 = nn.BatchNorm2d(256)
        self.conv1 = nn.Conv2d(in_channel_down, 256, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(in_channel_right, 256, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(256)

        self.conv_d1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv_d2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv_l = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(256 * 3, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

    def forward(self, left, down, right):
        left = F.relu(self.bn0(self.conv0(left)), inplace=True)  # 256 channels
        down = F.relu(self.bn1(self.conv1(down)), inplace=True)  # 256 channels
        right = F.relu(self.bn2(self.conv2(right)), inplace=True)  # 256

        down_1 = self.conv_d1(down)

        w1 = self.conv_l(left)

        if down.size()[2:] != left.size()[2:]:
            down_ = F.interpolate(down, size=left.size()[2:], mode='bilinear')
            z1 = F.relu(w1 * down_, inplace=True)
        else:
            z1 = F.relu(w1 * down, inplace=True)

        if down_1.size()[2:] != left.size()[2:]:
            down_1 = F.interpolate(down_1, size=left.size()[2:], mode='bilinear')

        z2 = F.relu(down_1 * left, inplace=True)

        # z3
        down_2 = self.conv_d2(right)
        if down_2.size()[2:] != left.size()[2:]:
            down_2 = F.interpolate(down_2, size=left.size()[2:], mode='bilinear')
        z3 = F.relu(down_2 * left, inplace=True)

        out = torch.cat((z1, z2, z3), dim=1)
        return F.relu(self.bn3(self.conv3(out)), inplace=True)

    def initialize(self):
        weight_init(self)

class SODNet1(nn.Module):
    def __init__(self, cfg):
        super(SODNet1, self).__init__()
        g_size = (2048, 256)
        l4_size = (1024, 256)
        l3_size = (512, 256)
        l2_size = (256, 256)

        self.cfg = cfg
        self.bkbone = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=14, scale=8)
        self.bkbone.initialize()

        self.g = nn.Sequential(
            nn.Conv2d(g_size[0], g_size[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.g4 = nn.Sequential(
            nn.Conv2d(g_size[0], g_size[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.g3 = nn.Sequential(
            nn.Conv2d(g_size[0], g_size[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.g2 = nn.Sequential(
            nn.Conv2d(g_size[0], g_size[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.l4 = nn.Sequential(
            nn.Conv2d(l4_size[0], l4_size[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.l3 = nn.Sequential(
            nn.Conv2d(l3_size[0], l3_size[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.l2 = nn.Sequential(
            nn.Conv2d(l2_size[0], l2_size[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.fam45 = Fusion(1024, 256, 256)
        self.fam34 = Fusion(512, 256, 256)
        self.fam23 = Fusion(256, 256, 256)

        self.srm_g = SR(256)
        self.srm_l4 = SR(256)
        self.srm_l3 = SR(256)
        self.srm_l2 = SR(256)

        self.linear_g = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.linear_l4 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.linear_l3 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.linear_l2 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)

        self.initialize()

    def forward(self, x):
        out1, out2, out3, out4, out5 = self.bkbone(x)
        g = self.g(out5)
        g4 = self.g4(out5)
        g3 = self.g3(out5)
        g2 = self.g2(out5)

        # l4 = self.l4(out4)
        # l3 = self.l3(out3)
        # l2 = self.l2(out2)

        srmg = self.srm_g(g4)

        out4 = self.fam45(out4, g, g4)
        out3 = self.fam34(out3, out4, g3)
        out2 = self.fam23(out2, out3, g2)

        # srmg_up = F.interpolate(srmg, size=l4.size()[2:], mode='bilinear')
        # srml4 = self.srm_l4(F.relu(srmg_up * l4, inplace=True))
        #
        # srml4_up = F.interpolate(srml4, size=l3.size()[2:], mode='bilinear')
        # srml3 = self.srm_l3(F.relu(srml4_up * l3, inplace=True))
        #
        # srml3_up = F.interpolate(srml3, size=l2.size()[2:], mode='bilinear')
        # srml2 = self.srm_l2(F.relu(srml3_up * l2, inplace=True))

        srml4 = self.srm_l4(out4)
        srml3 = self.srm_l4(out3)
        srml2 = self.srm_l4(out2)

        gout = self.linear_g(srmg)
        l4out = self.linear_l4(srml4)
        l3out = self.linear_l3(srml3)
        l2out = self.linear_l2(srml2)

        outg = F.interpolate(gout, size=x.size()[2:], mode='bilinear')
        outl4 = F.interpolate(l4out, size=x.size()[2:], mode='bilinear')
        outl3 = F.interpolate(l3out, size=x.size()[2:], mode='bilinear')
        outl2 = F.interpolate(l2out, size=x.size()[2:], mode='bilinear')

        return outg, outl4, outl3, outl2

    def initialize(self):
        if self.cfg.snapshot:
            try:
                self.load_state_dict(torch.load(self.cfg.snapshot))
            except:
                print("Warning: please check the snapshot file:", self.cfg.snapshot)
                pass
        else:
            weight_init(self)

class SODNet(nn.Module):
    def __init__(self, cfg):
        super(SODNet, self).__init__()
        self.cfg = cfg
        # self.bkbone  = ResNet()
        self.bkbone = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=14, scale=8)
        self.bkbone.initialize()

        self.hm3 = HM(2048, 2048)
        self.hm2 = HM(2048, 2048)
        self.hm1 = HM(2048, 2048)
        self.hm0 = HM(256, 2048)

        self.fu3 = Fusion(1024, 256, 256)
        self.fu2 = Fusion(512, 256, 256)
        self.fu1 = Fusion(256, 256, 256)

        self.srm5 = SR(256)
        self.srm4 = SR(256)
        self.srm3 = SR(256)
        self.srm2 = SR(256)

        self.linear5 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.linear4 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.linear3 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)
        self.linear2 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1)

        self.initialize()

    def forward(self, x):
        out1, out2, out3, out4, out5_ = self.bkbone(x)
        # GCF
        out4_a = self.hm3(out5_, out5_)
        out3_a = self.hm2(out5_, out5_)
        out2_a = self.hm1(out5_, out5_)
        # HA
        out5_a = self.sa55(out5_, out5_)
        out5 = self.hm0(out5_a, out5_)
        # out
        out5 = self.srm5(out5)
        out4 = self.srm4(self.fu3(out4, out5, out4_a))
        out3 = self.srm3(self.fu2(out3, out4, out3_a))
        out2 = self.srm2(self.fu1(out2, out3, out2_a))

        out5 = F.interpolate(self.linear5(out5), size=x.size()[2:], mode='bilinear')
        out4 = F.interpolate(self.linear4(out4), size=x.size()[2:], mode='bilinear')
        out3 = F.interpolate(self.linear3(out3), size=x.size()[2:], mode='bilinear')
        out2 = F.interpolate(self.linear2(out2), size=x.size()[2:], mode='bilinear')
        return out2, out3, out4, out5

    def initialize(self):
        if self.cfg.snapshot:
            try:
                self.load_state_dict(torch.load(self.cfg.snapshot))
            except:
                print("Warning: please check the snapshot file:", self.cfg.snapshot)
                pass
        else:
            weight_init(self)


def weight_init(module):
    for n, m in module.named_children():
        print('initialize: ' + n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            for mm in m:
                weight_init(mm)
        else:
            m.initialize()