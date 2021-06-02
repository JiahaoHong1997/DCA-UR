import torch
from torch import nn
from datetime import datetime
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

class Spatial(nn.Module):
    def __init__(self):
        super(Spatial, self).__init__()
        self.conv1 = nn.Conv2d(513, 1, 3, 1, 1)
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm2d(1)
        self._init_weight()

    def forward(self, cur_value, perv_predicted):
        cur_value1 = torch.cat([cur_value, perv_predicted], dim=1)
        pre_value = self.relu(self.batchnorm(self.conv1(cur_value1)))

        out = cur_value * pre_value

        # from tensorboardX import SummaryWriter
        # from torchvision.utils import make_grid
        # writer = SummaryWriter(comment=TIMESTAMP)
        #
        # writer.add_image('pre_value1',
        #                  make_grid(pre_value[0].detach().cpu().unsqueeze(dim=1), nrow=10, padding=1, normalize=True,
        #                            pad_value=1, scale_each=True, range=(0, 1)), 0)
        # writer.add_image('pre_value2',
        #                  make_grid(pre_value[1].detach().cpu().unsqueeze(dim=1), nrow=10, padding=1, normalize=True,
        #                            pad_value=1, scale_each=True, range=(0, 1)), 0)
        # writer.add_image('pre_value3',
        #                  make_grid(pre_value[2].detach().cpu().unsqueeze(dim=1), nrow=10, padding=1, normalize=True,
        #                            pad_value=1, scale_each=True, range=(0, 1)), 0)
        # writer.add_image('out1',
        #                  make_grid(out[0].detach().cpu().unsqueeze(dim=1), nrow=10, padding=1, normalize=True,
        #                            pad_value=1, scale_each=True, range=(0, 1)), 0)
        # writer.add_image('out2',
        #                  make_grid(out[1].detach().cpu().unsqueeze(dim=1), nrow=10, padding=1, normalize=True,
        #                            pad_value=1, scale_each=True, range=(0, 1)), 0)
        # writer.add_image('out3',
        #                  make_grid(out[2].detach().cpu().unsqueeze(dim=1), nrow=10, padding=1, normalize=True,
        #                            pad_value=1, scale_each=True, range=(0, 1)), 0)
        #
        # writer.flush()
        # writer.close()

        return out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)