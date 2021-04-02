import numpy as np
import torch


class FeatureBank:

    def __init__(self, obj_n, memory_budget, device):
        self.obj_n = obj_n
        self.device = device

        self.info = [None for _ in range(obj_n)]
        self.peak_n = np.zeros(obj_n)
        self.replace_n = np.zeros(obj_n)

    def init_bank(self, keys, values, frame_idx=0):

        self.keys = keys
        self.values = values

        for class_idx in range(self.obj_n):
            _, bank_n = keys[class_idx].shape   # bank_n = h*w  (h = H/16,w = W/16)
            # print('keys:',keys[class_idx].shape)   # (128, h*w)
            self.info[class_idx] = torch.zeros((bank_n, 2), device=self.device)
            self.info[class_idx][:, 0] = frame_idx
            self.peak_n[class_idx] = max(self.peak_n[class_idx], self.info[class_idx].shape[0])
            # print(self.peak_n[class_idx])

    def update(self, prev_key, prev_value, frame_idx):

        for class_idx in range(self.obj_n):

            self.keys[class_idx] = torch.cat([self.keys[class_idx], prev_key[class_idx]], dim=1)
            self.values[class_idx] = \
                torch.cat([self.values[class_idx], prev_value[class_idx]], dim=1)

            self.peak_n[class_idx] = max(self.peak_n[class_idx], self.info[class_idx].shape[0])

            self.info[class_idx][:, 1] = torch.clamp(self.info[class_idx][:, 1], 0, 1e5)  # Prevent inf
