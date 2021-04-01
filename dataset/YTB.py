# python原生库
import os
import random
import json
from glob import glob
import numpy as np
from itertools import compress

# pytorch库
import torch
from torch.utils import data
import torchvision.transforms as TF

# 自写库
import myutils
from transforms import transforms as mytrans

MAX_TRAINING_SKIP = 70

class YTB_train(data.Dataset):

    def __init__(self, root, output_size, dataset_file='meta.json', clip_n=6, max_obj_n=11, increment=1, max_skip=2):
        self.root = root
        self.clip_n = clip_n
        self.output_size = output_size
        self.max_obj_n = max_obj_n
        self.increment = increment
        self.max_skip = max_skip

        dataset_path = os.path.join(root, dataset_file)
        with open(dataset_path, 'r') as json_file:  # 读取json文件
            meta_data = json.load(json_file)

        self.dataset_list = list(meta_data['videos'])
        self.dataset_size = len(self.dataset_list)  # 读取视频数量

        # 图像变换
        self.random_horizontal_flip = mytrans.RandomHorizontalFlip(0.3)  # 以0.3的概率对图像和mask做水平翻转
        self.color_jitter = TF.ColorJitter(0.1, 0.1, 0.1, 0.02)  # 随机更改图像的亮度，对比度和饱和度
        # 保持中心不变的图像的随机仿射变换
        self.random_affine = mytrans.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.95, 1.05), shear=10)
        # 将给定的PIL图像裁剪为随机大小和纵横比
        self.random_resize_crop = mytrans.RandomResizedCrop(output_size, (0.3, 0.5), (0.95, 1.05))
        self.to_tensor = TF.ToTensor()  # 将PIL图像或np.array转化成tensor
        self.to_onehot = mytrans.ToOnehot(max_obj_n, shuffle=False)  # 将mask转换成onehot格式

    def __len__(self):
        return self.dataset_size

    def increase_max_skip(self):
        self.max_skip = min(self.max_skip + self.increment, MAX_TRAINING_SKIP)

    def set_max_skip(self, max_skip):
        self.max_skip = max_skip

    def __getitem__(self, idx):
        video_name = self.dataset_list[idx]  # 视频编号
        img_dir = os.path.join(self.root, 'JPEGImages', video_name)
        mask_dir = os.path.join(self.root, 'Annotations', video_name)

        img_list = sorted(glob(os.path.join(img_dir, '*.jpg')))  # 获得每个视频文件夹下各帧排序后的列表
        mask_list = sorted(glob(os.path.join(mask_dir, '*.png')))  # 获得每个视频mask文件夹下各帧排序后的列表

        # TODO:调整选取训练帧的策略，参考cfbi或STM原文中的训练策略
        # idx_list = list(range(len(img_list)))  # 视频帧的索引
        # idx0 = idx_list[0]
        # del idx_list[0]
        # random.shuffle(idx_list)  # 将idx_list中元素随机排序
        # idx_list = idx_list[:self.clip_n]  # 取出前clip_n个帧
        # idx_list[0] = idx0

        last_frame = -1
        nframes = len(img_list)
        idx_list = list()

        if nframes < self.clip_n:
            print("少于6帧的视频编号：", video_name)
            for i in range(0, nframes):
                idx_list.append(i)
            for i in range(0, self.clip_n-nframes):
                idx_list.append(nframes-1)
        else:
            for i in range(self.clip_n):
                if i == 0:
                    last_frame = random.sample(range(0, nframes - self.clip_n + 1), 1)[0]
                else:
                    last_frame = \
                    random.sample(range(last_frame + 1, min(last_frame + self.max_skip + 1, nframes - self.clip_n + i + 1)),
                                1)[0]
                idx_list.append(last_frame)

        frames = torch.zeros((self.clip_n, 3, self.output_size, self.output_size), dtype=torch.float)
        masks = torch.zeros((self.clip_n, self.max_obj_n, self.output_size, self.output_size), dtype=torch.float)
        # print('idx_list:', idx_list)

        for i, frame_idx in enumerate(idx_list):  # range: (0,clip_n)
            img = myutils.load_image_in_PIL(img_list[frame_idx], 'RGB')
            mask = myutils.load_image_in_PIL(mask_list[frame_idx], 'P')

            if i > 0:
                img = self.color_jitter(img)
                img, mask = self.random_affine(img, mask)

            roi_cnt = 0
            while roi_cnt < 10:
                img_roi, mask_roi = self.random_resize_crop(img, mask)
                mask_roi = np.array(mask_roi, np.uint8)  # np.uint8:无符号数（0到255）
                # print('mask_roi:',mask_roi[200:220,200:220])
                # print('mask_roi:',mask_roi.shape)

                # TODO:to_onehot方法可能导致每个sample的obj_n不等，可以修改这个方法，使每个sample的obj_n相等，用于多卡训练
                if i == 0:  # 将第一帧设置为reference
                    mask_roi, obj_list = self.to_onehot(mask_roi)  # 第一帧中所有的标注物体都会出现，所以需要一个obj_list来记录
                    obj_n = len(obj_list) + 1
                else:
                    mask_roi, _ = self.to_onehot(mask_roi, obj_list)  # 后面的帧中可能不会出现某一物体
                # print('mask_roi.onehot:',mask_roi[200:220,200:220])
                # print('mask_roi:', mask_roi.shape)

                if torch.any(mask_roi[0] == 0).item():
                    break

                roi_cnt += 1

            frames[i] = self.to_tensor(img_roi)
            masks[i] = mask_roi

        info = {
            'name': video_name,  # 视频编号
            'idx_list': idx_list  # 取出的帧的索引
        }

        return frames, masks[:, :obj_n], obj_n, info


class YouTube_Test(data.Dataset):

    def __init__(self, root, dataset_file='meta.json', output_size=(495, 880), max_obj_n=11):
        self.root = root
        self.max_obj_n = max_obj_n
        self.out_h, self.out_w = output_size

        dataset_path = os.path.join(root, dataset_file)
        with open(dataset_path, 'r') as json_file:
            self.meta_data = json.load(json_file)

        self.dataset_list = list(self.meta_data['videos'])
        self.dataset_size = len(self.dataset_list)

        self.to_tensor = TF.ToTensor()
        self.to_onehot = mytrans.ToOnehot(max_obj_n, shuffle=False)

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, idx):

        video_name = self.dataset_list[idx]

        img_dir = os.path.join(self.root, 'JPEGImages', video_name)
        mask_dir = os.path.join(self.root, 'Annotations', video_name)

        img_list = sorted(glob(os.path.join(img_dir, '*.jpg')))
        basename_list = [os.path.basename(x)[:-4] for x in img_list]
        video_len = len(img_list)
        selected_idx = np.ones(video_len, np.bool)

        objs = self.meta_data['videos'][video_name]['objects']
        obj_n = 1
        video_obj_appear_st_idx = video_len

        for obj_idx, obj_gt in objs.items():
            obj_n = max(obj_n, int(obj_idx) + 1)
            video_obj_appear_idx = basename_list.index(obj_gt['frames'][0])
            video_obj_appear_st_idx = min(video_obj_appear_st_idx, video_obj_appear_idx)

        selected_idx[:video_obj_appear_st_idx] = False
        selected_idx = selected_idx.tolist()

        img_list = list(compress(img_list, selected_idx))
        basename_list = list(compress(basename_list, selected_idx))

        video_len = len(img_list)
        obj_vis = np.zeros((video_len, obj_n), np.uint8)
        obj_vis[:, 0] = 1
        obj_st = np.zeros(obj_n, np.uint8)

        tmp_img = myutils.load_image_in_PIL(img_list[0], 'RGB')
        original_w, original_h = tmp_img.size
        if original_h < self.out_h:
            out_h, out_w = original_h, original_w
        else:
            out_h = self.out_h
            out_w = int(original_w / original_h * self.out_h)
        masks = torch.zeros((obj_n, out_h, out_w), dtype=torch.bool)

        basename_to_save = list()
        for obj_idx, obj_gt in objs.items():
            obj_idx = int(obj_idx)
            basename_to_save += obj_gt['frames']

            frame_idx = basename_list.index(obj_gt['frames'][0])
            obj_st[obj_idx] = frame_idx
            obj_vis[frame_idx:, obj_idx] = 1

            mask_path = os.path.join(mask_dir, obj_gt['frames'][0] + '.png')
            mask_raw = myutils.load_image_in_PIL(mask_path, 'P')
            mask_raw = mask_raw.resize((out_w, out_h))
            mask_raw = torch.from_numpy(np.array(mask_raw, np.uint8))

            masks[obj_idx, mask_raw == obj_idx] = 1

        basename_to_save = sorted(list(set(basename_to_save)))

        frames = torch.zeros((video_len, 3, out_h, out_w), dtype=torch.float)
        for i in range(video_len):
            img = myutils.load_image_in_PIL(img_list[i], 'RGB')
            img = img.resize((out_w, out_h))
            frames[i] = self.to_tensor(img)

        info = {
            'name': video_name,
            'num_frames': video_len,
            'obj_vis': obj_vis,
            'obj_st': obj_st,
            'basename_list': basename_list,
            'basename_to_save': basename_to_save,
            'original_size': (original_h, original_w)
        }

        return frames, masks, obj_n, info


if __name__ == '__main__':
    pass
