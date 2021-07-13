import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import os
import argparse
from PIL import Image
import torch
import torch.nn as nn
from torch.nn import functional as F

from dataset import DAVIS_Test, YouTube_Test
from model import TELG, FeatureBank, MaskBank
import myutils

torch.set_grad_enabled(False)


def get_args():
    parser = argparse.ArgumentParser(description='Eval TE-LG')
    parser.add_argument('--gpu', type=str,
                        help='GPU card id.')
    parser.add_argument('--level', type=int, default=1, required=True,
                        help='1: DAVIS17. 2: Youtube-VOS. ')
    parser.add_argument('--viz', action='store_true',
                        help='Visualize data.')
    parser.add_argument('--dataset', type=str, default=None, required=True,
                        help='Dataset folder.')
    parser.add_argument('--resume', type=str, required=True,
                        help='Path to the checkpoint (default: none)')
    parser.add_argument('--prefix', type=str,
                        help='Prefix to the model name.')
    parser.add_argument('--use_pre', action='store_true', 
                        help='If use previous frame.')
    parser.add_argument('--use_power', action='store_true',
                        help='If every frames for multi-object and every 5 frames for single-object.')
    return parser.parse_args()


def eval_DAVIS(model, model_name, dataloader):
    fps = myutils.FrameSecondMeter()

    for seq_idx, V in enumerate(dataloader):

        frames, masks, obj_n, info = V
        seq_name = info['name'][0]
        obj_n = obj_n.item()

        seg_dir = os.path.join('./output', model_name, seq_name)
        if not os.path.exists(seg_dir):
            os.makedirs(seg_dir)

        if args.viz:
            overlay_dir = os.path.join('./overlay', model_name, seq_name)
            if not os.path.exists(overlay_dir):
                os.makedirs(overlay_dir)

        frames, masks = frames[0].to(device), masks[0].to(device)
        H, W = frames.size(2), frames.size(3)
        frame_n = info['num_frames'][0].item()

        pred_mask = masks[0:1]
        pred = torch.argmax(pred_mask[0], dim=0).cpu().numpy().astype(np.uint8)
        seg_path = os.path.join(seg_dir, '00000.png')
        myutils.save_seg_mask(pred, seg_path, palette)

        if args.viz:
            overlay_path = os.path.join(overlay_dir, '00000.png')
            myutils.save_overlay(frames[0], pred, overlay_path, palette)


        fb = FeatureBank(obj_n)
        k4_list, v4_list, h, w = model.memorize(frames[0:1], pred_mask, 0, 0)
        keys_list = k4_list.copy()
        values_list = v4_list.copy()
        predforupdate = torch.zeros(1, obj_n, H, W).to(device)

        mb = MaskBank(obj_n)
        maskforbank = nn.functional.interpolate(pred_mask, size=(h, w), mode='bilinear', align_corners=True)
        maskforbank = maskforbank.view(obj_n, 1, -1)
        mask_list = [maskforbank[i] for i in range(obj_n)]

        prev_in_mem = True

        for t in tqdm(range(1, frame_n), desc=f'{seq_idx} {seq_name}'):

            fb.init_bank(keys_list, values_list)
            mb.init_bank(mask_list)

            if not prev_in_mem and args.use_pre:
                fb.update(k4_list, v4_list)
                mb.update(prev_list)

            score, _, f16 = model.segment(frames[t:t + 1], fb, mb, False)

            pred_mask = F.softmax(score, dim=1)
            pred1 = torch.argmax(pred_mask[0], dim=0)

            for j in range(obj_n):
                predforupdate[:, j] = (pred1 == j).long()

            pred = torch.argmax(pred_mask[0], dim=0).cpu().numpy().astype(np.uint8)
            # pred[pred!=3] = 0
            seg_path = os.path.join(seg_dir, f'{t:05d}.png')
            myutils.save_seg_mask(pred, seg_path, palette)

            # TODO: 如果前一帧要加入memory，则在此处更新key，value，mask；否则，在之后的if判断中考虑
            # k4_list, v4_list, _, _ = model.memorize(frames[t:t + 1], pred_mask, t, f16)
            # maskforbank = nn.functional.interpolate(predforupdate, size=(h, w), mode='bilinear', align_corners=True)
            # maskforbank = maskforbank.view(obj_n, 1, -1)
            # prev_list = [maskforbank[i] for i in range(obj_n)]

            if not args.use_power:
                if t < frame_n - 1 and t % 5 == 0:
                    k4_list, v4_list, _, _ = model.memorize(frames[t:t + 1], pred_mask, t, f16)
                    maskforbank = nn.functional.interpolate(predforupdate, size=(h, w), mode='bilinear',
                                                            align_corners=True)
                    maskforbank = maskforbank.view(obj_n, 1, -1)
                    prev_list = [maskforbank[i] for i in range(obj_n)]
                    for class_idx in range(obj_n):
                        keys_list[class_idx] = torch.cat([keys_list[class_idx], k4_list[class_idx]], dim=1)
                        values_list[class_idx] = torch.cat([values_list[class_idx], v4_list[class_idx]], dim=1)
                        mask_list[class_idx] = torch.cat([mask_list[class_idx], prev_list[class_idx]], dim=1)

                    prev_in_mem = True
                else:
                    prev_in_mem = False
            else:
                if obj_n == 2:  # single object
                    if t < frame_n - 1 and t % 5 == 0:
                        for class_idx in range(obj_n):
                            keys_list[class_idx] = torch.cat([keys_list[class_idx], k4_list[class_idx]], dim=1)
                            values_list[class_idx] = torch.cat([values_list[class_idx], v4_list[class_idx]], dim=1)
                            mask_list[class_idx] = torch.cat([mask_list[class_idx], prev_list[class_idx]], dim=1)

                        prev_in_mem = True
                else:  # multi-objects
                    if t < frame_n - 1 and t % 1 == 0:
                        for class_idx in range(obj_n):
                            keys_list[class_idx] = torch.cat([keys_list[class_idx], k4_list[class_idx]], dim=1)
                            values_list[class_idx] = torch.cat([values_list[class_idx], v4_list[class_idx]], dim=1)
                            mask_list[class_idx] = torch.cat([mask_list[class_idx], prev_list[class_idx]], dim=1)

                        prev_in_mem = True
                    else:
                        prev_in_mem = False

            if args.viz:
                overlay_path = os.path.join(overlay_dir, f'{t:05d}.png')
                myutils.save_overlay(frames[t], pred, overlay_path, palette)

        fps.add_frame_n(frame_n)

        fps.end()
        print(myutils.gct(), 'fps:', fps.fps)


def eval_YouTube(model, model_name, dataloader):
    seq_n = len(dataloader)
    fps = myutils.FrameSecondMeter()

    for seq_idx, V in enumerate(dataloader):

        frames, masks, obj_n, info = V

        frames, masks = frames[0].to(device), masks[0].to(device)
        H, W = frames.size(2), frames.size(3)
        frame_n = info['num_frames'][0].item()
        seq_name = info['name'][0]
        obj_n = obj_n.item()
        obj_st = [info['obj_st'][0, i].item() for i in range(obj_n)]
        basename_list = [info['basename_list'][i][0] for i in range(frame_n)]
        basename_to_save = [info['basename_to_save'][i][0] for i in range(len(info['basename_to_save']))]
        obj_vis = info['obj_vis'][0]
        original_size = (info['original_size'][0].item(), info['original_size'][1].item())

        seg_dir = os.path.join('./output', model_name, seq_name)
        if not os.path.exists(seg_dir):
            os.makedirs(seg_dir)

        if args.viz:
            overlay_dir = os.path.join('./overlay', model_name, seq_name)
            if not os.path.exists(overlay_dir):
                os.makedirs(overlay_dir)

        # Compose the first mask
        pred_mask = torch.zeros_like(masks).unsqueeze(0).float()
        for i in range(1, obj_n):
            if obj_st[i] == 0:
                pred_mask[0, i] = masks[i]
        pred_mask[0, 0] = 1 - pred_mask.sum(dim=1)

        pred_mask_output = F.interpolate(pred_mask, original_size)
        pred = torch.argmax(pred_mask_output[0], dim=0).cpu().numpy().astype(np.uint8)
        seg_path = os.path.join(seg_dir, basename_list[0] + '.png')
        myutils.save_seg_mask(pred, seg_path, palette)

        if args.viz:
            frame_out = F.interpolate(frames[0].unsqueeze(0), original_size).squeeze(0)
            overlay_path = os.path.join(overlay_dir, basename_list[0] + '.png')
            myutils.save_overlay(frame_out, pred, overlay_path, palette)

        fb = FeatureBank(obj_n)

        k4_list, v4_list, k4_h, k4_w = model.memorize(frames[0:1], pred_mask, 0)  # 参考帧
        fb.init_bank(k4_list, v4_list)

        mb = MaskBank(obj_n)
        maskforbank = nn.functional.interpolate(pred_mask, size=(k4_h, k4_w), mode='bilinear', align_corners=True)
        maskforbank = maskforbank.view(obj_n, 1, -1)
        mask_list = [maskforbank[i] for i in range(obj_n)]
        mb.init_bank(mask_list)
        predforupdate = torch.zeros(1, obj_n, H, W).to(device)

        for t in trange(1, frame_n, desc=f'{seq_idx:3d}/{seq_n:3d} {seq_name}'):

            score, _ = model.segment(frames[t:t + 1], fb, mb, False)

            reset_list = list()
            for i in range(1, obj_n):
                # If this object is invisible.
                if obj_vis[t, i] == 0:
                    score[0, i] = -1000

                # If this object appears, reset the score map
                if obj_st[i] == t:
                    reset_list.append(i)
                    score[0, i] = -1000
                    score[0, i][masks[i]] = 1000
                    for j in range(obj_n):
                        if j != i:
                            score[0, j][masks[i]] = -1000

            pred_mask = F.softmax(score, dim=1)
            pred1 = torch.argmax(pred_mask[0], dim=0)
            for j in range(obj_n):
                predforupdate[:, j] = (pred1 == j).long()

            if t < frame_n - 1:
                k4_list, v4_list, _, _ = model.memorize(frames[t:t + 1], pred_mask, t)
                maskforbank = nn.functional.interpolate(predforupdate, size=(k4_h, k4_w), mode='bilinear',
                                                            align_corners=True)
                maskforbank = maskforbank.view(obj_n, 1, -1)
                prev_list = [maskforbank[i] for i in range(obj_n)]
                if len(reset_list) > 0:
                    fb.init_bank(k4_list, v4_list)
                    mb.init_bank(prev_list)
                else:
                    fb.update(k4_list, v4_list)
                    mb.update(prev_list)

            if basename_list[t] in basename_to_save:
                pred_mask_output = F.interpolate(score, original_size)
                pred = torch.argmax(pred_mask_output[0], dim=0).cpu().numpy().astype(np.uint8)
                seg_path = os.path.join(seg_dir, basename_list[t] + '.png')
                myutils.save_seg_mask(pred, seg_path, palette)

                if args.viz:
                    frame_out = F.interpolate(frames[t].unsqueeze(0), original_size).squeeze(0)
                    overlay_path = os.path.join(overlay_dir, basename_list[t] + '.png')
                    myutils.save_overlay(frame_out, pred, overlay_path, palette)

        fps.add_frame_n(frame_n)

        fps.end()
        print(myutils.gct(), 'fps:', fps.fps)


def main():

    model = TELG(device=device, load_imagenet_params=False)
    model = model.to(device)
    model.eval()

    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            end_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'], strict=False)
            train_loss = checkpoint['loss']
            seed = checkpoint['seed']
            print(myutils.gct(),
                  f'Loaded checkpoint {args.resume}. (end_epoch: {end_epoch}, train_loss: {train_loss}, seed: {seed})')
        else:
            print(myutils.gct(), f'No checkpoint found at {args.resume}')
            raise IOError

    if args.level == 1:
        model_name = 'TELG_DAVIS17val'
        dataset = DAVIS_Test(args.dataset, '2017/val.txt')
    elif args.level == 2:
        model_name = 'TELG_YoutubeVOS'
        dataset = YouTube_Test(args.dataset)
    elif args.level == 3:
        model_name = 'TELG_DAVIS16val'
        dataset = DAVIS_Test(root=args.dataset, img_set='2016/val.txt', single_obj=True)
    else:
        raise ValueError(f'{args.level} is unknown.')

    if args.prefix:
        model_name += f'_{args.prefix}'
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    print(myutils.gct(), f'Model name: {model_name}')

    if args.level == 1:
        eval_DAVIS(model, model_name, dataloader)
    elif args.level == 2:
        eval_YouTube(model, model_name, dataloader)
    elif args.level == 3:
        eval_DAVIS(model, model_name, dataloader)
    else:
        raise ValueError(f'{args.level} is unknown.')


if __name__ == '__main__':
    args = get_args()
    print(myutils.gct(), 'Args =', args)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    device = torch.device('cuda', 0)
    
    palette = Image.open(os.path.join(args.dataset, 'mask_palette.png')).getpalette()

    main()

    print(myutils.gct(), 'Evaluation done.')
