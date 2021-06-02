import torch


class MaskBank:

    def __init__(self, obj_n):

        self.obj_n = obj_n
        self.mask_list = list()

    def init_bank(self, mask_list):   # mask:[obj_n]list, size:(1, H*W)

        self.mask_list = mask_list.copy()

    def update(self, pre_mask_list):

        for class_idx in range(self.obj_n):
            self.mask_list[class_idx] = torch.cat([self.mask_list[class_idx], pre_mask_list[class_idx]], dim=1)


