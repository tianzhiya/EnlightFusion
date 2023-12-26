from torch import nn
import torch.nn.functional as F


class SIM(nn.Module):

    def __init__(self, norm_nc, label_nc, nhidden=64):
        super().__init__()

        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False, track_running_stats=False)

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Sequential(
            nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(num_features=norm_nc)

    def forward(self, x, segmap):
        normalizedX = self.param_free_norm(x)
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='bilinear')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = self.bn(normalizedX * (1 + gamma)) + beta
        return out
