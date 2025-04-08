import torch

class HeatmapLoss(torch.nn.Module):
    """
    loss for detection heatmap
    """
    def __init__(self):
        super(HeatmapLoss, self).__init__()

    def forward(self, pred, gt):
        l = ((pred - gt)**2)
        l = l.mean(dim=3).mean(dim=2).mean(dim=1)
        return l ## l of dim bsize
class AdaptiveWingLoss(torch.nn.Module):
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1):
        super(AdaptiveWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, pred, target):
        '''
        :param pred: BxNxHxH
        :param target: BxNxHxH
        :return:
        '''

        '''y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.theta]
        delta_y2 = delta_y[delta_y >= self.theta]
        y1 = y[delta_y < self.theta]
        y2 = y[delta_y >= self.theta]
        loss1 = self.omega * torch.log(1 + torch.pow(delta_y1 / self.omega, self.alpha - y1))
        A = self.omega * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))) * (self.alpha - y2) * (
            torch.pow(self.theta / self.epsilon, self.alpha - y2 - 1)) * (1 / self.epsilon)
        C = self.theta * A - self.omega * torch.log(1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))
        loss2 = A * delta_y2 - C
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))
        '''
        delta_y = (target - pred).abs()

        # 使用张量比较运算
        mask_low = delta_y < self.theta
        mask_high = ~mask_low

        loss = torch.zeros_like(delta_y)

        # 低误差区域计算（使用**替代.pow()）
        y_low = target[mask_low]
        delta_low = delta_y[mask_low]
        loss_low = self.omega * torch.log(1 + (delta_low / self.omega) ** (self.alpha - y_low))
        loss[mask_low] = loss_low

        # 高误差区域计算（统一使用张量运算）
        y_high = target[mask_high]
        delta_high = delta_y[mask_high]
        ratio = self.theta / self.epsilon
        exponent = self.alpha - y_high

        A = self.omega / (1 + ratio ** exponent) \
            * (self.alpha - y_high) * (ratio ** (exponent - 1)) / self.epsilon
        C = self.theta * A - self.omega * torch.log(1 + ratio ** exponent)
        loss_high = A * delta_high - C
        loss[mask_high] = loss_high

        return loss.mean(dim=[1, 2, 3])