import torch
from torch import nn
from models.layers import Conv, Hourglass, Pool, Residual
from task.loss import HeatmapLoss
from torchvision.models import resnet50

class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(-1, 256, 4, 4)

class Merge(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)
    
class PoseNet(nn.Module):
    def __init__(self, nstack, inp_dim, oup_dim, bn=False, increase=0, **kwargs):
        super(PoseNet, self).__init__()
        
        self.nstack = nstack

        resnet = resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-2])
        # 上采样模块：将分辨率从 1/32 恢复到 1/4
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1),  # 1/32 -> 1/16
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),  # 1/16 -> 1/8
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(512, inp_dim, kernel_size=4, stride=2, padding=1),  # 1/8 -> 1/4
            nn.BatchNorm2d(inp_dim),
            nn.ReLU(inplace=True)
        )
        # self.pre = nn.Sequential(
        #     Conv(3, 64, 7, 2, bn=True, relu=True),
        #     Residual(64, 128),
        #     Pool(2, 2),
        #     Residual(128, 128),
        #     Residual(128, inp_dim)
        # )
        
        self.hgs = nn.ModuleList( [
        nn.Sequential(
            Hourglass(4, inp_dim, bn, increase),
        ) for i in range(nstack)] )
        
        self.features = nn.ModuleList( [
        nn.Sequential(
            Residual(inp_dim, inp_dim),
            Conv(inp_dim, inp_dim, 1, bn=True, relu=True)
        ) for i in range(nstack)] )
        
        self.outs = nn.ModuleList( [Conv(inp_dim, oup_dim, 1, relu=False, bn=False) for i in range(nstack)] )
        self.merge_features = nn.ModuleList( [Merge(inp_dim, inp_dim) for i in range(nstack-1)] )
        self.merge_preds = nn.ModuleList( [Merge(oup_dim, inp_dim) for i in range(nstack-1)] )
        self.nstack = nstack
        self.heatmapLoss = HeatmapLoss()

    def forward(self, imgs):
        ## our posenet
        x = imgs.permute(0, 3, 1, 2) #x of size 1,3,inpdim,inpdim
        #x = self.pre(x)
        x = self.resnet_features(x)
        x = self.upsample(x)
        combined_hm_preds = []
        for i in range(self.nstack):
            hg = self.hgs[i](x)
            feature = self.features[i](hg)
            preds = self.outs[i](feature)
            combined_hm_preds.append(preds)
            if i < self.nstack - 1:
                x = x + self.merge_preds[i](preds) + self.merge_features[i](feature)
        return torch.stack(combined_hm_preds, 1)

    def calc_loss(self, combined_hm_preds, heatmaps):
        combined_loss = []
        for i in range(self.nstack):
            combined_loss.append(self.heatmapLoss(combined_hm_preds[0][:,i], heatmaps))
        combined_loss = torch.stack(combined_loss, dim=1)
        return combined_loss
