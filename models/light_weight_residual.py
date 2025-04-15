from torch import nn

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size - 1) // 2, bias=True)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ConvBNReLU(nn.Sequential):#depthwise conv+BN+relu6，用于构建InvertedResidual。
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        #参数：输入的tensor的通道数，输出通道数，卷积核大小，卷积步距，输入与输出对应的块数（改为输入的层数就是depth wise conv了，详见https://blog.csdn.net/weixin_43572595/article/details/110563397）
        padding = (kernel_size - 1) // 2#根据卷积核大小获取padding大小
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            #构建depthwise conv卷积，不使用偏置，因为后面有BN
            nn.BatchNorm2d(out_channel),#BN层，输入参数为输入的tensor的层数
            nn.ReLU6(inplace=True)#relu6激活函数，inplace原地操作tensor减少内存占用
        )

class InvertedResidual(nn.Module):#逆向残差结构
    def __init__(self, inp_dim, out_dim, stride=1, expand_ratio=6):
        #参数：输入的tensor的通道数，输出的通道数，中间depthwise conv的步距，第一个1x1普通卷积的channel放大倍数
        super(InvertedResidual, self).__init__()
        hidden_channel = inp_dim * expand_ratio#隐层的输入通道数，对应中间depthwise conv的输入通道数
        self.use_shortcut = stride == 1 and inp_dim == out_dim
        #使用shortcut的条件：输入与输出shape一样，即没有缩放（stride=1）,维度一样（in_channel = out_channel）

        layers = []#搜集各个层
        if expand_ratio != 1:#这个是由于第一个卷积没有维度放大，因而不需要第一个1x1普通卷积
            # 1x1 pointwise conv，第一个普通1x1卷积，升维
            layers.append(ConvBNReLU(inp_dim, hidden_channel, kernel_size=1))
        layers.extend([
            # 3x3 depthwise conv
            ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            # 1x1 pointwise conv(linear)第二个普通1x1卷积，降维
            nn.Conv2d(hidden_channel, out_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_dim),#BN
        ])

        self.conv = nn.Sequential(*layers)#将上面的集成到一起

    def forward(self, x):#前向传播过程，直接用上面弄好的conv层
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)


class SepConv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride = 1, bn = False, relu = True):
        super(SepConv, self).__init__()
        self.inp_dim = inp_dim
        self.deepthwise_conv = nn.Conv2d(inp_dim,
                                         inp_dim,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=(kernel_size-1)//2,
                                         bias=True,
                                         groups=inp_dim)
        self.pointwise_conv = nn.Conv2d(inp_dim,
                                        out_dim,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        bias=True,
                                        groups=1)
        #self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=True)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)
    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.deepthwise_conv(x)
        x = self.pointwise_conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class DSResidual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(DSResidual, self).__init__()
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim / 2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv2 = SepConv(int(out_dim / 2), out_dim, 3, relu=False)
        # self.bn3 = nn.BatchNorm2d(int(out_dim / 2))
        # self.conv3 = Conv(int(out_dim / 2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        # out = self.bn3(out)
        # out = self.relu(out)
        # out = self.conv3(out)
        out += residual
        return out

# Here is the code :

import torch
import torch.nn as nn
import torch.nn.functional as F


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape: b, num_channels, h, w  -->  b, groups, channels_per_group, h, w
    x = x.view(batchsize, groups, channels_per_group, height, width)

    # channelshuffle
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class ShuffleNetResidual(nn.Module):
    expansion = 1

    def __init__(self, in_dim, out_dim, stride=1, groups=4):
        super(ShuffleNetResidual, self).__init__()

        mid_channels = out_dim // 4
        self.stride = stride
        if in_dim == 24:
            self.groups = 1
        else:
            self.groups = groups
        self.GConv1 = nn.Sequential(
            nn.Conv2d(in_dim, mid_channels, kernel_size=1, stride=1, groups=self.groups, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        self.DWConv = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=self.stride, padding=1, groups=self.groups, bias=False),
            nn.BatchNorm2d(mid_channels)
        )

        self.GConv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_dim, kernel_size=1, stride=1, groups=self.groups, bias=False),
            nn.BatchNorm2d(out_dim)
        )

        if self.stride == 2:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.GConv1(x)
        out = channel_shuffle(out, groups=self.groups)
        out = self.DWConv(out)
        out = self.GConv2(out)
        short = self.shortcut(x)
        if self.stride == 2:
            out = F.relu(torch.cat([out, short], dim=1))
        else:
            out = F.relu(out + short)
        return out


