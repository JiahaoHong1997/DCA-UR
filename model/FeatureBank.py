import torch


class FeatureBank:

    def __init__(self, obj_n, h, w):
        self.obj_n = obj_n
        self.keys = torch.zeros(1, h, w)
        self.values = list()

    def init_bank(self, keys, values):

        self.keys = keys.clone()
        self.values = values.copy()

    def update(self, prev_key, prev_value):

        self.keys = torch.cat([self.keys, prev_key], dim=2)
        for class_idx in range(self.obj_n):

            self.values[class_idx] = \
                torch.cat([self.values[class_idx], prev_value[class_idx]], dim=1)
