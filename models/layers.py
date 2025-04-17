from torch import nn
from models.CBAM import CBAM
from models.CBAM import ChannelAttention
from models.CBAM import SpatialAttention
Pool = nn.MaxPool2d

def batchnorm(x):
    return nn.BatchNorm2d(x.size()[1])(x)

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride = 1, bn = False, relu = True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=True)
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
class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim, down=false):
        super(Residual, self).__init__()
        self.relu = nn.ReLU()
        stride = 1
        if down:
            stride = 2
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim/2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim/2))
        self.conv2 = SepConv(int(out_dim/2), out_dim, 3, stride=stride,relu=False)
        # self.bn3 = nn.BatchNorm2d(int(out_dim/2))
        # self.conv3 = Conv(int(out_dim/2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, stride=stride,relu=False)
        self.attention = ChannelAttention(out_dim)
        if inp_dim == out_dim and down == false:
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
        out = self.attention(out)
        return out 

class Hourglass(nn.Module):
    def __init__(self, n, f, bn=None, increase=0):
        super(Hourglass, self).__init__()
        nf = f + increase
        self.attention = SpatialAttention()
        self.up1 = nn.Sequential(
                Residual(nf, nf),
            )
        # Lower branch
        ##self.pool1 = Pool(2, 2)
        self.low1 = Residual(f, nf, true)
        self.n = n
        # Recursive hourglass
        if self.n > 1:
            self.low2 = Hourglass(n-1, nf, bn=bn)
        else:
            self.low2 = nn.Sequential(
                Residual(nf, nf)
            )
        self.low3 = Residual(nf, f)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        ##self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        # self.up2 = nn.Sequential(
        #     nn.ConvTranspose2d(
        #         in_channels=f,
        #         out_channels=f,
        #         kernel_size=4,
        #         stride=2,
        #         padding=1
        #     ),
        #     nn.BatchNorm2d(f),
        #     nn.ReLU(inplace=True)
        # )
    def forward(self, x):
        up1  = self.up1(x)
        ##pool1 = self.pool1(x)
        low1 = self.low1(x)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2  = self.up2(low3)
        return self.attention(up1 + up2)
