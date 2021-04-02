import os
import time
import argparse
import numpy as np
from tqdm import tqdm
from glob import glob

import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch.utils import data
from tensorboardX import SummaryWriter

from model import STM
from model import FeatureBank
from model import MaskBank
from dataset import YTB_train, DAVIS17_Train, PreTrain
import myutils


def get_args():
    parser = argparse.ArgumentParser(description='Train STM_refinement')
    parser.add_argument('--gpu', type=str, help="0; 0,1; 0,3; etc", required=True)
    parser.add_argument('--dataset', type=str, default=None, required=False, help='Dataset floder.')
    parser.add_argument('--seed', type=int, default=-1, help='Random seed.')
    parser.add_argument('--log', action='store_true', help='Save the training results.')
    parser.add_argument('--level', type=int, default=0, help='0: pretrain. 1: DAVIS. 2: Youtube-VOS.')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate, default 1e-5.')
    parser.add_argument('--resume', type=str, help='Path to the checkpoint (default: none)')
    parser.add_argument('--new', action='store_true', help='Train the model from the begining.')
    parser.add_argument('--scheduler-step', type=int, default=30, help='Scheduler step size. Default 25.')
    parser.add_argument('--total-epochs', type=int, default=120, help='Total running epochs. Default 100.')
    parser.add_argument('--obj-n', type=int, default=3,
                        help='Max number of objects that will be trained at the same time.')
    parser.add_argument('--clip-n', type=int, default=6, help='Max frames that will be sampled as a batch.')
    parser.add_argument('--budget', type=int, default=1000000,
                        help='Max number of features that feature bank can store. Default: 300000')
    parser.add_argument('--epochs_per_increment', type=int, default=20, help='Max epochs per increment occurs')
    parser.add_argument('--lu', type=float, default=0.5, help='Regularization factor, default 0.5.')

    return parser.parse_args()


def run_pretrain(model, dataloader, criterion, optimizer):
    stats = myutils.AvgMeter()
    uncertainty_stats = myutils.AvgMeter()

    progress_bar = tqdm(dataloader, desc='Main Train')
    for iter_idx, sample in enumerate(progress_bar):

        frames, masks, obj_n, info = sample

        obj_n = obj_n.item()
        if obj_n == 1:
            continue

        frames, masks = frames[0].to(device), masks[0].to(device)

        fb_global = FeatureBank(obj_n, args.budget, device)
        k4_list, v4_list, _, _ = model.memorize(frames[0:1], masks[0:1])
        fb_global.init_bank(k4_list, v4_list)

        mb = MaskBank(obj_n, device)
        maskforbank = nn.functional.interpolate(masks[0:1], size=(25, 25), mode='bilinear', align_corners=True)
        mb.init_bank(maskforbank)  # maskForBank: (1,obj_n,25,25)

        scores, uncertainty = model.segment(frames[1:], fb_global, mb)
        label = torch.argmax(masks[1:], dim=1).long()
        optimizer.zero_grad()
        loss = criterion(scores, label)
        loss = loss + args.lu * uncertainty
        loss.backward()
        optimizer.step()

        uncertainty_stats.update(uncertainty.item())
        stats.update(loss.item())
        progress_bar.set_postfix(loss=f'{loss.item():.5f} ({stats.avg:.5f} {uncertainty_stats.avg:.5f})')

    progress_bar.close()

    return stats.avg


def run_maintrain(model, dataloader, criterion, optimizer):
    stats = myutils.AvgMeter()
    uncertainty_stats = myutils.AvgMeter()

    progress_bar = tqdm(dataloader, desc='Main Train')
    for iter_idx, sample in enumerate(progress_bar):

        frames, masks, obj_n, info = sample

        obj_n = obj_n.item()
        if obj_n == 1:
            continue

        frames, masks = frames[0].to(device), masks[0].to(device)

        fb_global = FeatureBank(obj_n, args.budget, device)
        k4_list, v4_list, _, _ = model.memorize(frames[0:1], masks[0:1])
        fb_global.init_bank(k4_list, v4_list)

        mb = MaskBank(obj_n, device)
        maskforbank = nn.functional.interpolate(masks[0:1], size=(25, 25), mode='bilinear', align_corners=True)
        mb.init_bank(maskforbank)   # maskForBank: (1,obj_n,25,25)

        frame_n, _, H, W = frames.size()
        scores = torch.zeros(frame_n - 1, obj_n, H, W).to(device)
        uncertainties = torch.zeros(frame_n - 1).to(device)

        for t in range(1, frame_n):
            score, uncertainty = model.segment(frames[t:t + 1], fb_global, mb, info)  # 1 , obj_n , H , W
            scores[t - 1] = score[0]
            uncertainties[t - 1] = uncertainty

            if t < frame_n - 1:
                k4_list, v4_list, _, _ = model.memorize(frames[t:t + 1], masks[t:t + 1])
                fb_global.update(k4_list, v4_list, t)
                maskforupdate = nn.functional.interpolate(score, size=(25, 25), mode='bilinear', align_corners=True)
                mb.update(maskforupdate)

        # print('score:',scores.size())
        label = torch.argmax(masks[1:], dim=1).long()  # frame_idx , H , W
        # print('label:',label.size())

        uncertainty = uncertainties.mean()
        optimizer.zero_grad()
        loss = criterion(scores, label)
        loss = loss + args.lu * uncertainty
        loss.backward()
        optimizer.step()

        uncertainty_stats.update(uncertainty.item())
        stats.update(loss.item())
        progress_bar.set_postfix(loss=f'{loss.item():.5f} ({stats.avg:.5f} {uncertainty_stats.avg:.5f})')

    progress_bar.close()

    return stats.avg


def main():
    if args.level == 0:
        dataset = PreTrain(args.dataset, output_size=400, clip_n=args.clip_n, max_obj_n=args.obj_n)
    elif args.level == 1:
        dataset = DAVIS17_Train(args.dataset, output_size=400, clip_n=args.clip_n, max_obj_n=args.obj_n)
    elif args.level == 2:
        dataset = YTB_train(args.dataset, output_size=400, clip_n=args.clip_n, max_obj_n=args.obj_n)
    elif args.level == 3:
        dataset1 = DAVIS17_Train('../STM/DataSets/DAVIS', output_size=400, clip_n=args.clip_n, max_obj_n=args.obj_n)
        dataset2 = YTB_train('../STM/DataSets/YTB/train', output_size=400, clip_n=args.clip_n, max_obj_n=args.obj_n)
        dataset = torch.utils.data.ConcatDataset([dataset1, dataset2])
    else:
        raise ValueError(f'{args.level} is unknown.')

    # batch sizes
    train_batch_size = num_devices

    dataloader = data.DataLoader(dataset, batch_size=train_batch_size, shuffle=True, num_workers=2, pin_memory=True)
    print(myutils.gct(), f'Load level {args.level} dataset: {len(dataset)} training cases.')

    model = STM(device=device, load_imagenet_params=True)
    print("Model instantiated")

    if num_devices > 1:
        model = nn.DataParallel(model)
        print("Model parallelised in {} GPUs".format(num_devices))

    model = model.to(device)
    print("Model sent to cuda")
    model.train()
    model.apply(myutils.set_bn_eval)  # turn-off BN

    # parameters, optmizer and loss
    params = model.parameters()
    optimizer = torch.optim.AdamW(filter(lambda x: x.requires_grad, params), args.lr)

    criterion = torch.nn.CrossEntropyLoss().to(device)

    start_epoch = 0
    best_loss = 100000000
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model'], strict=False)
            seed = checkpoint['seed']
            if args.level != 0 and args.new is False:
                skips = checkpoint['max_skip']
                if isinstance(skips, list):
                    for idx, skip in enumerate(skips):
                        dataset.datasets[idx].set_max_skip(skip)
                else:
                    dataset.set_max_skip(skips)

            if not args.new:
                start_epoch = checkpoint['epoch'] + 1
                optimizer.load_state_dict(checkpoint['optimizer'])
                best_loss = checkpoint['loss']
                print(myutils.gct(),
                      f'Loaded checkpoint {args.resume} (epoch: {start_epoch - 1}, best loss: {best_loss})')
            else:
                if args.seed < 0:
                    seed = int(time.time())
                else:
                    seed = args.seed
                print(myutils.gct(), f'Loaded checkpoint {args.resume}. Train from the beginning.')
        else:
            print(myutils.gct(), f'No checkpoint found at {args.resume}')
            raise IOError
    else:

        if args.seed < 0:
            seed = int(time.time())
        else:
            seed = args.seed

    print(myutils.gct(), 'Random seed:', seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    scheduler = torch.optim.lr_scheduler.StepLR(  # 动态调整学习率，每过25个epoch，学习率衰减一半
        optimizer, step_size=args.scheduler_step, gamma=0.5, last_epoch=start_epoch - 1)

    progress_bar = tqdm(range(start_epoch, args.total_epochs), desc='epoches')
    for epoch in progress_bar:

        lr = scheduler.get_last_lr()[0]  # 获得上一个epoch的学习率

        if args.level != 0:
            loss = run_maintrain(model, dataloader, criterion, optimizer)
        else:
            loss = run_pretrain(model, dataloader, criterion, optimizer)
        vis_writer.add_scalar('train/loss', loss, epoch)
        vis_writer.add_scalar('train/lr', lr)

        skips = 0

        if args.level != 0:
            if (epoch + 1) % args.epochs_per_increment == 0:
                if isinstance(dataset, data.ConcatDataset):
                    for dst in dataset.datasets:
                        dst.increase_max_skip()
                    skips = [ds.max_skip for ds in dataset.datasets]
                else:
                    dataset.increase_max_skip()
                    skips = dataset.max_skip

        print('')
        print(myutils.gct(), f'Epoch: {epoch} lr: {lr} max_skip:{skips}')

        if args.log:

            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': loss,
                'seed': seed,
                'max_skip': skips,
            }

            checkpoint_path = f'{model_path}/final.pth'
            torch.save(checkpoint, checkpoint_path)

            checkpoint_path = f'{model_path}/epoch_{epoch:03d}.pth'
            torch.save(checkpoint, checkpoint_path)

            if best_loss > loss:
                best_loss = loss

                checkpoint_path = f'{model_path}/best.pth'
                torch.save(checkpoint, checkpoint_path)

                print('Best model updated.')

        scheduler.step()
    progress_bar.close()


if __name__ == '__main__':

    args = get_args()
    print(myutils.gct(), f'Args = {args}')

    MODEL = 'STM_refinement'
    GPU = args.gpu
    print(MODEL, ', Using Dataset:', args.dataset)
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU
    # Device infos
    if torch.cuda.is_available():
        device = torch.device("cuda")
        num_devices = torch.cuda.device_count()
        print('Cuda version: ', torch.version.cuda)
        print('Current GPU id: ', torch.cuda.current_device())
        print('Device name: ', torch.cuda.get_device_name(device=torch.cuda.current_device()))
        print('Number of available devices:', num_devices)
    else:
        raise ValueError('CUDA is required. --gpu must be used.')

    if args.log:
        if not os.path.exists('logs'):
            os.makedirs('logs')

        prefix = f'level{args.level}'
        log_dir = 'log/{}'.format(time.strftime(prefix + '_%Y%m%d-%H%M%S'))
        log_path = os.path.join(log_dir, 'log')
        model_path = os.path.join(log_dir, 'model')
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        myutils.save_scripts(log_dir, scripts_to_save=glob('*.*'))
        myutils.save_scripts(log_dir, scripts_to_save=glob('dataset/*.py', recursive=True))
        myutils.save_scripts(log_dir, scripts_to_save=glob('model/*.py', recursive=True))
        myutils.save_scripts(log_dir, scripts_to_save=glob('myutils/*.py', recursive=True))

        vis_writer = SummaryWriter(log_path)

        print(myutils.gct(), f'Creat log dir: {log_dir}')

        main()

        if args.log:
            vis_writer.close()

        print(myutils.gct(), 'Training done.')
