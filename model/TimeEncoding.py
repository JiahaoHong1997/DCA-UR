import torch
from torch import nn
import numpy as np


class TimeEncoding(nn.Module):

    def __init__(self, d_hid, nframe=50, nposition=900):
        super(TimeEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('time_table', self._get_sinusoid_encoding_table(nframe, d_hid))
        self.register_buffer('pos_table', self._get_exp_encoding_table(nposition, d_hid))

    def _get_sinusoid_encoding_table(self, nframe, d_hid):
        def get_time_angle_vec(position):
            return [position / np.power(10000, 2 * hid_j / (d_hid // 3)) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_time_angle_vec(pos_i) for pos_i in range(nframe)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) / 100  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) / 100  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0).transpose(1, 2)  # (1,dim,n_position)

    def _get_exp_encoding_table(self, nposition, d_hid):
        def get_position_angle_vec(position):
            return [-1 * position * np.power(10000, 2 * hid_j / (d_hid // 3)) for hid_j in range(d_hid)]

        exp_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(nposition)])
        exp_table[:, 0::2] = np.exp(exp_table[:, 0::2]) / 10
        exp_table[:, 1::2] = np.exp(exp_table[:, 1::2]) / 10

        return torch.FloatTensor(exp_table).unsqueeze(0).transpose(1, 2)

    def forward(self, x, idx):
        x = x + self.time_table[:, :, idx:idx + 1].expand(-1, -1, x.size(-1)).clone().detach()
        return x + self.pos_table[:, :, :x.size(-1)].clone().detach()
