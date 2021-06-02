import os
import numpy as np
import random
from glob import glob

import torch
from torch.utils import data
import torchvision.transforms as TF

from transforms import transforms as mytrans
import myutils
MAX_TRAINING_SKIP = 25

class BL30K(data.Dataset):

    def __init__(self, root, output_size, clip_n=3, max_obj_n=11, increment=5, max_skip=10):
        self.root = root
        self.clip_n = clip_n
        self.output_size = output_size
        self.max_obj_n = max_obj_n
        self.increment = increment
        self.max_skip = max_skip

        video_num = 0
        dataset_path = os.path.join(root, 'Annotations')
        for lists in os.listdir(dataset_path):
            sub_path = os.path.join(dataset_path, lists)
            if os.path.isdir(sub_path):
                video_num = video_num+1

        self.video_num = video_num

        self.dataset_list = os.listdir(dataset_path)

        print("self.dataset_list", len(self.dataset_list))
        print("self.video_num", self.video_num)
        self.img_list = list()
        self.mask_list = list()

        self.random_horizontal_flip = mytrans.RandomHorizontalFlip(0.3)
        self.color_jitter = TF.ColorJitter(0.1, 0.1, 0.1, 0.03)
        self.random_affine = mytrans.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10)
        self.random_resize_crop = mytrans.RandomResizedCrop(output_size, (0.8, 1))
        self.to_tensor = TF.ToTensor()
        self.to_onehot = mytrans.ToOnehot(max_obj_n, shuffle=True)

    def __len__(self):
        return self.video_num

    def increase_max_skip(self):
        self.max_skip = min(self.max_skip + self.increment, MAX_TRAINING_SKIP)

    def set_max_skip(self, max_skip):
        self.max_skip = max_skip

    def __getitem__(self, idx):

        video_name = self.dataset_list[idx]
        img_dir = os.path.join(self.root, 'JPEGImages', video_name)
        mask_dir = os.path.join(self.root, 'Annotations', video_name)

        img_list = sorted(glob(os.path.join(img_dir, '*.jpg')))
        mask_list = sorted(glob(os.path.join(mask_dir, '*.png')))

        last_frame = -1
        nframes = len(img_list)
        idx_list = list()

        for i in range(self.clip_n):
            if i == 0:
                last_frame = random.sample(range(0, nframes - self.clip_n + 1), 1)[0]

            else:
                last_frame = random.sample(
                    range(last_frame + 1, min(last_frame + self.max_skip + 1, nframes - self.clip_n + i + 1)),
                    1)[0]

            idx_list.append(last_frame)

        frames = torch.zeros((self.clip_n, 3, self.output_size, self.output_size), dtype=torch.float)
        masks = torch.zeros((self.clip_n, self.max_obj_n, self.output_size, self.output_size), dtype=torch.float)

        for i, frame_idx in enumerate(idx_list):

            img = myutils.load_image_in_PIL(img_list[frame_idx], 'RGB')
            mask = myutils.load_image_in_PIL(mask_list[frame_idx], 'P')

            if i > 0:
                img = self.color_jitter(img)
                img, mask = self.random_affine(img, mask)

            roi_cnt = 0
            while roi_cnt < 10:
                img_roi, mask_roi = self.random_resize_crop(img, mask)

                mask_roi = np.array(mask_roi, np.uint8)

                if i == 0:
                    mask_roi, obj_list = self.to_onehot(mask_roi)
                    obj_n = len(obj_list) + 1
                else:
                    mask_roi, _ = self.to_onehot(mask_roi, obj_list)

                if torch.any(mask_roi[0] == 0).item():
                    break

                roi_cnt += 1

            frames[i] = self.to_tensor(img_roi)
            masks[i] = mask_roi


        info = {
            'name': video_name,
            'idx_list': idx_list
        }

        return frames, masks[:, :obj_n], obj_n, info

