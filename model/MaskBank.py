import torch
import torch.nn.functional as NF


class MaskBank:

    def __init__(self, obj_n, device):

        self.obj_n = obj_n
        self.device = device
        self.mask_list = list()

    def init_bank(self, mask):   # mask:[obj_n]list, size:(1, H*W)

        mask = mask.view(self.obj_n, 1, -1)
        mask_list = [mask[i] for i in range(self.obj_n)]
        self.mask_list = mask_list

    def update(self, pre_mask):

        pre_mask = pre_mask.view(self.obj_n, 1, -1)
        pre_mask_list = [pre_mask[i] for i in range(self.obj_n)]
        for class_idx in range(self.obj_n):
            self.mask_list[class_idx] = torch.cat([self.mask_list[class_idx], pre_mask_list[class_idx]], dim=1)


