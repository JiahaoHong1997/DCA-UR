import torch
from torch import nn


class Spatial(nn.Module):
    def __init__(self):
        super(Spatial, self).__init__()
        self.conv1 = nn.Conv2d(1025, 1, 3, 1, 1)
        self.sigmoid = nn.Sigmoid()
        self._init_weight()

    def forward(self, cur_value, perv_predicted):
        cur_value1 = torch.cat([cur_value, perv_predicted], dim=1)

        pre_value = self.sigmoid(self.conv1(cur_value1))

        out = cur_value * pre_value

        return out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)