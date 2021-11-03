import torch


class FeatureBank:

    def __init__(self, obj_n, h, w):
        self.obj_n = obj_n
        self.keys = torch.zeros(1, h, w)
        self.values = list()

    def init_keys(self, keys):
        self.keys = keys.clone()

    def init_values(self, values):
        self.values = values.copy()

    def update_keys(self, keys):
        self.keys = keys.clone()

    def update_values(self, prev_value):
        for class_idx in range(self.obj_n):
            self.values[class_idx] = \
                torch.cat([self.values[class_idx], prev_value[class_idx]], dim=1)
