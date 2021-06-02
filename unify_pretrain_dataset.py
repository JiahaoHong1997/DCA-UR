import os
from os import path
import argparse
import numpy as np
import cv2
import multiprocessing
from shutil import copyfile
from glob import glob
from tqdm import tqdm
from PIL import Image

import myutils

try:
    from pycocotools.coco import COCO
except ImportError as e:
    print(e)


def get_args():
    parser = argparse.ArgumentParser(description='Unify Pretrain Dataset')
    parser.add_argument('--dst', type=str, default='./DataSets/pretrainDataset')
    parser.add_argument('--palette', type=str, default='./dataset/mask_palette.png')
    parser.add_argument('--worker', type=int, default=10, help='Threads number.')
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--src', type=str, required=True)
    return parser.parse_args()


def cp_files(src_list, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for src_path in tqdm(src_list, desc='cp'):
        dst_path = os.path.join(dst_dir, os.path.basename(src_path))
        copyfile(src_path, dst_path)


def cvt_mask_palette(data):
    src_path, dst_dir = data

    mask = cv2.imread(src_path)
    mask_size = mask.shape[:2]

    label = np.asarray(mask).reshape(-1, 3)
    obj_labels = list(set(map(tuple, label)))
    obj_labels.sort()

    new_label = np.zeros(label.shape[0], np.uint8)

    obj_cnt = 0
    for idx, label_id in enumerate(obj_labels):
        tmp = int(label_id[0]) + int(label_id[1]) + int(label_id[2])
        if 0 < tmp / 3 < 230:
            continue

        new_label[(label == label_id).all(axis=1)] = obj_cnt
        if obj_cnt != 1:
            obj_cnt += 1


    new_label = Image.fromarray(new_label.reshape(mask_size))
    new_label.putpalette(mask_palette)

    dst_path = os.path.join(dst_dir, os.path.basename(src_path))
    new_label.save(dst_path)


def cvt_MSRA10K():
    img_list = sorted(glob(os.path.join(args.src, 'MSRA10K_Imgs_GT/Imgs/', '*.jpg')), key=lambda x: (len(x), x))
    mask_list = sorted(glob(os.path.join(args.src, 'MSRA10K_Imgs_GT/Imgs/', '*.png')), key=lambda x: (len(x), x))

    dst_img_dir = os.path.join(args.dst, 'JPEGImages', args.name)
    dst_mask_dir = os.path.join(args.dst, 'Annotations', args.name)
    cp_files(img_list, dst_img_dir)

    if not os.path.exists(dst_mask_dir):
        os.makedirs(dst_mask_dir)

    mask_list = [(x, dst_mask_dir) for x in mask_list]
    pools = multiprocessing.Pool(16)
    pools.map(cvt_mask_palette, mask_list)
    pools.close()
    pools.join()


def cvt_ECSSD():
    img_list = sorted(glob(os.path.join(args.src, 'ecssd', '*.jpg')), key=lambda x: (len(x), x))
    mask_list = sorted(glob(os.path.join(args.src, 'ecssd', '*.png')), key=lambda x: (len(x), x))

    dst_img_dir = os.path.join(args.dst, 'JPEGImages', args.name)
    dst_mask_dir = os.path.join(args.dst, 'Annotations', args.name)
    cp_files(img_list, dst_img_dir)

    if not os.path.exists(dst_mask_dir):
        os.makedirs(dst_mask_dir)

    mask_list = [(x, dst_mask_dir) for x in mask_list]
    pools = multiprocessing.Pool(worker_n)
    pools.map(cvt_mask_palette, mask_list)
    pools.close()
    pools.join()


def cvt_PASCALS():
    img_list = sorted(glob(os.path.join(args.src, 'datasets/imgs/pascal', '*.jpg')), key=lambda x: (len(x), x))
    mask_list = sorted(glob(os.path.join(args.src, 'datasets/masks/pascal', '*.png')), key=lambda x: (len(x), x))

    dst_img_dir = os.path.join(args.dst, 'JPEGImages', args.name)
    dst_mask_dir = os.path.join(args.dst, 'Annotations', args.name)
    cp_files(img_list, dst_img_dir)

    if not os.path.exists(dst_mask_dir):
        os.makedirs(dst_mask_dir)

    mask_list = [(x, dst_mask_dir) for x in mask_list]
    pools = multiprocessing.Pool(worker_n)
    pools.map(cvt_mask_palette, mask_list)
    pools.close()
    pools.join()

def cvt_BIG():
    img_list = sorted(glob(os.path.join(args.src, 'BIG_small', '*.jpg')), key=lambda x: (len(x), x))
    mask_list = sorted(glob(os.path.join(args.src, 'BIG_small', '*.png')), key=lambda x: (len(x), x))

    dst_img_dir = os.path.join(args.dst, 'JPEGImages', args.name)
    dst_mask_dir = os.path.join(args.dst, 'Annotations', args.name)
    cp_files(img_list, dst_img_dir)

    if not os.path.exists(dst_mask_dir):
        os.makedirs(dst_mask_dir)

    mask_list = [(x, dst_mask_dir) for x in mask_list]
    pools = multiprocessing.Pool(16)
    pools.map(cvt_mask_palette, mask_list)
    pools.close()
    pools.join()

def cvt_HRSOD():
    img_list = sorted(glob(os.path.join(args.src, 'HRSOD_small', '*.jpg')), key=lambda x: (len(x), x))
    mask_list = sorted(glob(os.path.join(args.src, 'HRSOD_small', '*.png')), key=lambda x: (len(x), x))


    dst_img_dir = os.path.join(args.dst, 'JPEGImages', args.name)
    dst_mask_dir = os.path.join(args.dst, 'Annotations', args.name)
    cp_files(img_list, dst_img_dir)

    if not os.path.exists(dst_mask_dir):
        os.makedirs(dst_mask_dir)

    mask_list = [(x, dst_mask_dir) for x in mask_list]
    pools = multiprocessing.Pool(16)
    pools.map(cvt_mask_palette, mask_list)

    pools.close()
    pools.join()

def cvt_dut_te():
    img_list = sorted(glob(os.path.join(args.src, 'DUTS-TE', '*.jpg')), key=lambda x: (len(x), x))
    mask_list = sorted(glob(os.path.join(args.src, 'DUTS-TE', '*.png')), key=lambda x: (len(x), x))

    dst_img_dir = os.path.join(args.dst, 'JPEGImages', args.name)
    dst_mask_dir = os.path.join(args.dst, 'Annotations', args.name)
    cp_files(img_list, dst_img_dir)

    if not os.path.exists(dst_mask_dir):
        os.makedirs(dst_mask_dir)

    mask_list = [(x, dst_mask_dir) for x in mask_list]
    pools = multiprocessing.Pool(16)
    pools.map(cvt_mask_palette, mask_list)
    pools.close()
    pools.join()

def cvt_dut_tr():
    img_list = sorted(glob(os.path.join(args.src, 'DUTS-TR', '*.jpg')), key=lambda x: (len(x), x))
    mask_list = sorted(glob(os.path.join(args.src, 'DUTS-TR', '*.png')), key=lambda x: (len(x), x))

    dst_img_dir = os.path.join(args.dst, 'JPEGImages', args.name)
    dst_mask_dir = os.path.join(args.dst, 'Annotations', args.name)
    cp_files(img_list, dst_img_dir)

    if not os.path.exists(dst_mask_dir):
        os.makedirs(dst_mask_dir)

    mask_list = [(x, dst_mask_dir) for x in mask_list]
    pools = multiprocessing.Pool(16)
    pools.map(cvt_mask_palette, mask_list)
    pools.close()
    pools.join()

def cvt_fss():

    img_list = []
    mask_list = []
    classes = os.listdir(args.src)
    for c in classes:
        imgs = os.listdir(path.join(args.src, c))
        jpg_list = [im for im in imgs if 'jpg' in im[-3:].lower()]
        joint_list = [path.join(args.src, c, im) for im in jpg_list]
        img_list.extend(joint_list)
        img_list = sorted(img_list, key=lambda x: (len(x), x))

        png_list = [im for im in imgs if 'png' in im[-3:].lower()]
        joint_list2 = [path.join(args.src, c, im) for im in png_list]
        mask_list.extend(joint_list2)
        mask_list = sorted(mask_list, key=lambda x: (len(x), x))


    dst_img_dir = os.path.join(args.dst, 'JPEGImages', args.name)
    dst_mask_dir = os.path.join(args.dst, 'Annotations', args.name)
    cp_files(img_list, dst_img_dir)

    if not os.path.exists(dst_mask_dir):
        os.makedirs(dst_mask_dir)

    mask_list = [(x, dst_mask_dir) for x in mask_list]
    pools = multiprocessing.Pool(16)
    pools.map(cvt_mask_palette, mask_list)
    pools.close()
    pools.join()


def create_COCO_img_mask(data):
    img_id, dst_img_dir, dst_mask_dir = data

    img_info = coco.loadImgs(img_id)[0]
    h = img_info['height']
    w = img_info['width']

    # mask
    mask_all = np.zeros((h, w), np.uint8)
    anno_ids = coco.getAnnIds(imgIds=img_info['id'])
    anno_list = coco.loadAnns(anno_ids)

    obj_cnt = 1
    for idx, anno in enumerate(anno_list):
        if anno['area'] < 500:
            continue
        mask = coco.annToMask(anno)
        mask_all[mask > 0] = mask[mask > 0] * obj_cnt
        obj_cnt += 1

    if obj_cnt > 1:
        mask_all = Image.fromarray(mask_all)
        mask_all.putpalette(mask_palette)

        img_name = img_info['file_name'][:-4] + '.png'
        dst_path = os.path.join(dst_mask_dir, img_name)
        mask_all.save(dst_path)

        # img
        tmp = img_info['coco_url'].split('/')[-2:]
        img_path_src = os.path.join(args.src, tmp[0], tmp[1])
        img_path_dst = os.path.join(dst_img_dir, img_info['file_name'])
        copyfile(img_path_src, img_path_dst)


def cvt_COCO():
    global coco

    dst_img_dir = os.path.join(args.dst, 'JPEGImages', args.name)
    dst_mask_dir = os.path.join(args.dst, 'Annotations', args.name)

    if not os.path.exists(dst_img_dir):
        os.makedirs(dst_img_dir)
    if not os.path.exists(dst_mask_dir):
        os.makedirs(dst_mask_dir)

    # val2017
    anno_file = os.path.join(args.src, 'annotations', 'instances_val2017.json')
    print('Annotation path:', anno_file)
    coco = COCO(anno_file)  # Global var must be initialized before multiprocessing.Pool
    img_list = coco.getImgIds()
    img_list = [(x, dst_img_dir, dst_mask_dir) for x in img_list]
    pools = multiprocessing.Pool(worker_n)
    pools.imap(create_COCO_img_mask, img_list, chunk_size)
    pools.close()
    pools.join()
    print('val_over')

    # Train 2017
    anno_file = os.path.join(args.src, 'annotations', 'instances_train2017.json')
    print('Annotation path:', anno_file)
    coco = COCO(anno_file)  # Global var must be initialized before multiprocessing.Pool
    img_list = coco.getImgIds()
    print('1')
    img_list = [(x, dst_img_dir, dst_mask_dir) for x in img_list]
    print('2')
    pools = multiprocessing.Pool(worker_n)
    print('3')
    pools.imap(create_COCO_img_mask, img_list, chunk_size)
    print('4')
    pools.close()
    print('5')
    pools.join()
    print('train_over')


def cvt_mask_palette_VOC(data):
    src_path, dst_path = data
    mask = np.array(myutils.load_image_in_PIL(src_path, 'P'))
    mask[mask > 20] = 0
    mask = Image.fromarray(mask)
    mask.putpalette(mask_palette)

    mask.save(dst_path)


def cvt_VOC2012():
    dst_img_dir = os.path.join(args.dst, 'JPEGImages', args.name)
    dst_mask_dir = os.path.join(args.dst, 'Annotations', args.name)

    img_set = os.path.join(args.src, 'ImageSets/Segmentation', 'trainval.txt')
    img_path_list = list()
    mask_path_list = list()
    with open(os.path.join(img_set), 'r') as lines:
        for line in lines:
            img_name = line.strip()
            img_path_list.append(os.path.join(args.src, 'JPEGImages', img_name + '.jpg'))
            dst_mask_path = os.path.join(dst_mask_dir, img_name + '.png')
            mask_path_list.append((
                os.path.join(args.src, 'SegmentationObject', img_name + '.png'),
                dst_mask_path
            ))

    if not os.path.exists(dst_mask_dir):
        os.makedirs(dst_mask_dir)

    pools = multiprocessing.Pool(worker_n)
    pools.imap(cvt_mask_palette_VOC, mask_path_list, chunk_size)
    pools.close()
    pools.join()

    cp_files(img_path_list, dst_img_dir)


if __name__ == '__main__':
    args = get_args()

    mask_palette = Image.open(args.palette).getpalette()
    worker_n = args.worker
    chunk_size = 100
    coco = None

    # Ming-Ming Cheng, Niloy J Mitra, Xiaolei Huang, Philip HSTorr, and Shi-Min Hu. Global contrast based salient regiondetection.
    # Download: https://mmcheng.net/msra10k/
    if args.name == 'MSRA10K':
        cvt_MSRA10K()
        print(1)

    # Jianping Shi, Qiong Yan, Li Xu, and Jiaya Jia.  Hierar-chical image saliency detection on extended cssd
    # Download http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html
    # --name ECSSD --src /Ship01/Dataset/ECSSD
    if args.name == 'ECSSD':
        cvt_ECSSD()

    # Yin Li, Xiaodi Hou, Christof Koch, James M Rehg, and Alan L Yuille. The secrets of salient object segmentation.
    # Download: http://cbs.ic.gatech.edu/salobj
    # --name PASCAL-S --src /Ship01/Dataset/PASCAL-S
    if args.name == 'PASCAL-S':
        cvt_PASCALS()

    # harath Hariharan,  Pablo Arbel ́aez,  Lubomir Bourdev,Subhransu Maji, and Jitendra Malik.  Semantic contoursfrom inverse detectors.
    # Download http://cocodataset.org/#download
    # pycocotools: https://github.com/cocodataset/cocoapi/tree/master/PythonAPI
    # --name COCO --src /Ship01/Dataset/COCO
    if args.name == 'COCO':
        cvt_COCO()

    # Mark Everingham, Luc Van Gool, Christopher KI Williams,John Winn, and Andrew Zisserman. The pascal visual objectclasses (voc) challenge
    # Download http://host.robots.ox.ac.uk/pascal/VOC/voc2012/
    # --name PASCALVOC2012 --src /Ship01/Dataset/PASCALVOC2012
    if args.name == 'PASCALVOC2012':
        cvt_VOC2012()
    
    if args.name == 'BIG':

        cvt_BIG() 

    if args.name == 'HRSOD':
        cvt_HRSOD()

    if args.name == 'DUTS-TE':
        cvt_dut_te()

    if args.name == 'DUTS-TR':
        cvt_dut_tr()

    if args.name == 'fss':
        cvt_fss()