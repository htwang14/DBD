import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0, activate_before_residual=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.drop_rate = drop_rate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=True) or None
        self.activate_before_residual = activate_before_residual

    def forward(self, x):
        if not self.equalInOut and self.activate_before_residual == True:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class CDBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0, activate_before_residual=False, attn_var=1):
        super(CDBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.drop_rate = drop_rate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=True) or None
        self.activate_before_residual = activate_before_residual

        # original OE:
        if attn_var == 1:
            self.attn = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), 
                nn.Conv2d(out_planes, out_planes//16, kernel_size=1),
                nn.ReLU(True),
                nn.Conv2d(out_planes//16, out_planes, kernel_size=1),
                nn.Sigmoid()
            )
        # var2:
        elif attn_var == 2:
            self.attn = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), 
                nn.Conv2d(out_planes, out_planes, kernel_size=1, groups=out_planes),
                nn.Sigmoid()
            )
        # var3:
        elif attn_var == 3:
            self.attn = nn.Sequential(
                nn.AdaptiveAvgPool2d(1), 
                nn.Conv2d(out_planes, out_planes, kernel_size=1),
                nn.Sigmoid()
            )

    def forward(self, x):
        '''
        x: shape=(N,C,H,W)
        m: shape=(N,C)
        '''
        if not self.equalInOut and self.activate_before_residual == True:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.bn2(self.conv1(out if self.equalInOut else x))
        out = self.relu2(out)
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        masks = self.attn(out)
        out = out * masks
        out = torch.add(x if self.equalInOut else self.convShortcut(x), out)
        return out, masks


class CDWideResNet(nn.Module):
    def __init__(self, num_classes, depth=28, widen_factor=2, drop_rate=0.0, attn_var=1, partial=1):
        super(CDWideResNet, self).__init__()
        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        self.attn_var = attn_var
        self.partial = partial
        self.widen_factor = widen_factor
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1, bias=True)
        # 1st block
        self.block1 = self._make_layer(n, channels[0], channels[1], 1, drop_rate, activate_before_residual=True)
        # 2nd block
        self.block2 = self._make_layer(n, channels[1], channels[2], 2, drop_rate)
        # 3rd block
        self.block3 = self._make_cd_layer(n, channels[2], channels[3], 2, drop_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(channels[3])
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.fc = nn.Linear(channels[3], num_classes)
        self.channels = channels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()

    def _make_layer(self, nb_layers, in_planes, out_planes, stride, drop_rate=0, activate_before_residual=False):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(BasicBlock(i == 0 and in_planes or out_planes, out_planes,
                                i == 0 and stride or 1, drop_rate, activate_before_residual))
        return nn.Sequential(*layers)

    def _make_cd_layer(self, nb_layers, in_planes, out_planes, stride, drop_rate=0, activate_before_residual=False):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(CDBasicBlock(i == 0 and in_planes or out_planes, out_planes,
                                i == 0 and stride or 1, drop_rate, activate_before_residual, self.attn_var))
        return torch.nn.ModuleList(layers)

    def forward(self, x):
        
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)

        mask_per_layer = []
        for layer in self.block3:
            out, mask = layer(out)
            mask = mask.squeeze()
            if self.partial<1:
                # print(mask.shape)
                mask_per_layer.append(mask[:,0:int(64*self.widen_factor*self.partial)])
            else:
                mask_per_layer.append(mask)
        mask = torch.cat(mask_per_layer, dim=1)

        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(-1, self.channels)
        logits = self.fc(out)

        return logits

def CDWRN28(num_classes=10, widen_factor=2):
    return CDWideResNet(depth=28, num_classes=num_classes, widen_factor=widen_factor)