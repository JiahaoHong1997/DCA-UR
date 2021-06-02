import torch


class FeatureBank:

    def __init__(self, obj_n):
        self.obj_n = obj_n

    def init_bank(self, keys, values):

        self.keys = keys.copy()
        self.values = values.copy()

    def update(self, prev_key, prev_value):

        for class_idx in range(self.obj_n):

            self.keys[class_idx] = torch.cat([self.keys[class_idx], prev_key[class_idx]], dim=1)
            self.values[class_idx] = \
                torch.cat([self.values[class_idx], prev_value[class_idx]], dim=1)
