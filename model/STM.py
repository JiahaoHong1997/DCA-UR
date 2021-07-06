import math
import time
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50

import myutils
# from myutils.data import ToCuda
from datetime import datetime
from model import Position

TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())


class EncoderQ(nn.Module):
    def __init__(self, load_imagenet_params):
        super(EncoderQ, self).__init__()
        resnet = resnet50(pretrained=load_imagenet_params)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1  # 1/4, 256
        self.res3 = resnet.layer2  # 1/8, 512
        self.res4 = resnet.layer3  # 1/16, 1024

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, in_f):
        f = (in_f - self.mean) / self.std

        x = self.conv1(f)
        x = self.bn1(x)
        r1 = self.relu(x)
        x = self.maxpool(r1)
        r2 = self.res2(x)
        r3 = self.res3(r2)
        r4 = self.res4(r3)

        return r4, r3, r2, r1


class EncoderM(nn.Module):
    def __init__(self, load_imagenet_params):
        super(EncoderM, self).__init__()
        self.conv1_m = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1_o = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        resnet = resnet50(pretrained=load_imagenet_params)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1  # 1/4, 256
        self.res3 = resnet.layer2  # 1/8, 512
        self.res4 = resnet.layer3  # 1/16, 1024

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, in_f, in_m, in_o):
        f = (in_f - self.mean) / self.std

        x = self.conv1(f) + self.conv1_m(in_m) + self.conv1_o(in_o)
        x = self.bn1(x)
        r1 = self.relu(x)
        x = self.maxpool(r1)

        r2 = self.res2(x)
        r3 = self.res3(r2)
        r4 = self.res4(r3)
        return r4, r1


class KeyVaule(nn.Module):
    def __init__(self, indim, keydim, vauledim):
        super(KeyVaule, self).__init__()
        self.keydim = keydim
        self.vauledim = vauledim

        self.convkey = nn.Conv2d(indim, keydim, kernel_size=(3, 3), stride=1, padding=(1, 1))
        self.convvaule = nn.Conv2d(indim, vauledim, kernel_size=(3, 3), stride=1, padding=(1, 1))

    def forward(self, x):
        key = self.convkey(x)
        h, w = key.size(2), key.size(3)
        key = key.view(*key.shape[:2], -1)  # obj_n, key_dim, pixel_n

        vaule = self.convvaule(x)
        vaule = vaule.view(*vaule.shape[:2], -1)  # obj_n, vaule_dim, pixel_n
        return key, vaule, h, w


class Matcher(nn.Module):
    def __init__(self, top_k, km):
        super(Matcher, self).__init__()
        self.top_k = top_k
        self.km = km

    def forward(self, feature_bank, q_in, q_out, h, w, mask_bank, info):
        mem_out_list = []

        for i in range(0, feature_bank.obj_n):
            d_key, bank_n = feature_bank.keys[i].size()  # 128 , t*h*w

            # print('feature_bank.keys[i]:',feature_bank.keys[i].size())
            # print('feature_bank.keys[i].transpose(0, 1):',feature_bank.keys[i].transpose(0, 1).size())  # t*h*w , 128
            bs, _, n = q_in.size()
            p = torch.matmul(feature_bank.keys[i].transpose(0, 1), q_in) / math.sqrt(d_key)  # bs, t*h*w, h*w

            if self.km is not None:
                # Make a bunch of Gaussian distributions
                argmax_idx = p.max(2)[1]
                y_idx, x_idx = argmax_idx // w, argmax_idx % w
                g = myutils.make_gaussian(y_idx, x_idx, h, w, sigma=self.km)
                g = g.view(bs, bank_n, n)

                p = myutils.softmax_w_g_top(p, top=self.top_k, gauss=g) # bs, t*h*w, h*w
            else:
                if self.top_k is not None:
                    p = myutils.softmax_w_g_top(p, top=self.top_k, gauss=None)  # bs, t*h*w, h*w
                else:
                    p = F.softmax(p, dim=1)   # bs, t*h*w, h*w

            mem = torch.matmul(feature_bank.values[i], p)  # frame_idx, 512, h*w
            # print("mask_bank.mask_list[i]:",mask_bank.mask_list[i].size())
            # print("p:",p.size())
            mask_mem = torch.matmul(mask_bank.mask_list[i], p)  # 1, 1, h*w
            q_out_with_mask = q_out * mask_mem

            # from tensorboardX import SummaryWriter
            # from torchvision.utils import make_grid
            # writer = SummaryWriter(comment=TIMESTAMP)
            #
            # mmem = mem.reshape(mem.size()[0], mem.size()[1], h, w)
            # curr = q_out.reshape(q_out.size()[0], q_out.size()[1], h, w)
            # currWithMask = q_out_with_mask.reshape(q_out.size()[0], q_out.size()[1], h, w)
            # mmask_mem = mask_mem.reshape(mask_mem.size()[0], mask_mem.size()[1], h, w)
            #
            # writer.add_image('mem',
            #           make_grid(mmem[0].detach().cpu().unsqueeze(dim=1), nrow=20, padding=1, normalize=True,
            #                     pad_value=1, scale_each=True, range=(0, 1)), i)
            # writer.add_image('curr',
            #           make_grid(curr[0].detach().cpu().unsqueeze(dim=1), nrow=20, padding=1, normalize=True,
            #                     pad_value=1, scale_each=True, range=(0, 1)), i)
            # writer.add_image('currWithMask',
            #           make_grid(currWithMask[0].detach().cpu().unsqueeze(dim=1), nrow=20, padding=1, normalize=True,
            #                     pad_value=1, scale_each=True, range=(0, 1)), i)
            # writer.add_image('mmask_mem',
            #           make_grid(mmask_mem[0].detach().cpu().unsqueeze(dim=1), nrow=20, padding=1, normalize=True,
            #                     pad_value=1, scale_each=True, range=(0, 1)), i)
            #
            # writer.flush()
            # writer.close()

            mem_out_list.append(torch.cat([mem, q_out_with_mask], dim=1))  # frame_idx, 1024, h*w

        mem_out_tensor = torch.stack(mem_out_list, dim=0).transpose(0, 1)  # frame_idx, obj_n, 1024, h*w
        # print('mem_out_tensor:',mem_out_tensor.size())

        return mem_out_tensor


class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None, stride=1):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim and stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)

        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)

    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))

        if self.downsample is not None:
            x = self.downsample(x)

        return x + r


class Refine(nn.Module):
    def __init__(self, in_c, out_c, scale_factor=2):
        super(Refine, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.Res1 = ResBlock(out_c, out_c)
        self.Res2 = ResBlock(out_c, out_c)
        self.scale_factor = scale_factor

    def forward(self, high_level_feature, low_level_feature):
        f = self.conv(high_level_feature)
        s = self.Res1(f)
        m = s + F.interpolate(low_level_feature, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        m = self.Res2(m)
        return m


class Decoder(nn.Module):
    def __init__(self, device):
        super(Decoder, self).__init__()
        self.device = device
        self.hidden_dim = 256
        self.convFM = nn.Conv2d(1024, self.hidden_dim, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.ResMM = ResBlock(self.hidden_dim, self.hidden_dim)
        self.RF3 = Refine(512, self.hidden_dim)
        self.RF2 = Refine(self.hidden_dim, self.hidden_dim)

        self.pred2 = nn.Conv2d(self.hidden_dim, 2, kernel_size=(3, 3), padding=(1, 1), stride=1)

        # Local
        local_size = 7
        mdim_local = 32
        self.local_avg = nn.AvgPool2d(local_size, stride=1, padding=local_size // 2)
        self.local_max = nn.MaxPool2d(local_size, stride=1, padding=local_size // 2)
        self.local_convFM = nn.Conv2d(128, mdim_local, kernel_size=3, padding=1, stride=1)
        self.local_ResMM = ResBlock(mdim_local, mdim_local)
        self.local_pred2 = nn.Conv2d(mdim_local, 2, kernel_size=3, padding=1, stride=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, r4, r3, r2, r1=None, feature_shape=None):
        # r4:(frame_idx * obj_n,1024,H/16,W/16)
        # r3:(frame_idx * obj_n,512,H/8,W/8)
        # r2:(frame_idx * obj_n,256,H/4,W/4)
        m4 = self.ResMM(self.convFM(r4))  # 1024 -> hidden_dim  1/16
        m3 = self.RF3(r3, m4)  # hidden_dim  1/8
        m2 = self.RF2(r2, m3)  # hidden_dim  1/4

        p2 = self.pred2(F.relu(m2))  # hidden_dim -> 2
        # print('p2:',p2.size())   # frame_idx * obj_n, 2 , H/4 , W/4
        p = F.interpolate(p2, scale_factor=2, mode='bilinear', align_corners=False)  # 1
        # print('p:',p.size())   # frame_idx * obj_n , 2 , H , W

        bs, obj_n, h, w = feature_shape
        rough_seg = F.softmax(p, dim=1)[:, 1]
        rough_seg = rough_seg.view(bs, obj_n, h, w)
        rough_seg = F.softmax(rough_seg, dim=1)  # object-level normalization

        # Local refinement
        uncertainty = myutils.calc_uncertainty(rough_seg)
        uncertainty = uncertainty.expand(-1, obj_n, -1, -1).reshape(bs * obj_n, 1, h, w)

        # zero = torch.zeros_like(uncertainty)
        # one = torch.ones_like(uncertainty)
        # a = torch.where(uncertainty>0.8,one,uncertainty)
        # a = torch.where(a<=0.8,zero,uncertainty)  # 1,1,h,w


        rough_seg = rough_seg.view(bs * obj_n, 1, h, w)  # bs*obj_n, 1, h, w
        r1_weighted = r1 * rough_seg
        r1_local = self.local_max(r1_weighted)  # bs*obj_n, 64, h, w
        r1_local = r1_local / (self.local_max(rough_seg) + 1e-8)  # neighborhood reference
        r1_conf = self.local_avg(rough_seg)  # bs*obj_n, 1, h, w
        #
        # from tensorboardX import SummaryWriter
        # from torchvision.utils import make_grid
        # writer = SummaryWriter(comment=TIMESTAMP)
        #
        # for i in range (0,obj_n):
        #
        #     writer.add_image('uncertainty',
        #                 make_grid(uncertainty[i].detach().cpu().unsqueeze(dim=1), nrow=20, padding=1, normalize=True,
        #                             pad_value=1, scale_each=True, range=(0, 1)), i)
        #     writer.add_image('a',
        #                 make_grid(a[i].detach().cpu().unsqueeze(dim=1), nrow=20, padding=1, normalize=True,
        #                             pad_value=1, scale_each=True, range=(0, 1)), i)
        #     writer.add_image('rough_seg',
        #                 make_grid(rough_seg[i].detach().cpu().unsqueeze(dim=1), nrow=20, padding=1, normalize=True,
        #                             pad_value=1, scale_each=True, range=(0, 1)), i)
        #
        #     writer.add_image('r1',
        #                 make_grid(r1[i].detach().cpu().unsqueeze(dim=1), nrow=20, padding=1, normalize=True,
        #                             pad_value=1, scale_each=True, range=(0, 1)), i)
        # time.sleep(1)
        # writer.flush()
        # writer.close()

        # r1_local = r1_local*0
        local_match = torch.cat([r1, r1_local], dim=1)
        q = self.local_ResMM(self.local_convFM(local_match))
        q = r1_conf * self.local_pred2(F.relu(q))

        p = p + uncertainty * q
        p = F.interpolate(p, scale_factor=2, mode='bilinear', align_corners=False)

        p = F.softmax(p, dim=1)[:, 1]
        # print('p:',p.size())  # frame_idx * obj_n , 2 , H , W

        return p


class STM(nn.Module):
    def __init__(self, device, load_imagenet_params=False, top_k=50, km=None):
        super(STM, self).__init__()

        self.device = device
        self.encoder_q = EncoderQ(load_imagenet_params)
        self.encoder_m = EncoderM(load_imagenet_params)

        self.keyvalue_r4 = KeyVaule(indim=1024, keydim=128, vauledim=512)
        self.position = Position.PositionalEncoding(d_hid=1024)

        self.matcher = Matcher(top_k=top_k, km=km)

        self.decoder = Decoder(device)

    def memorize(self, frame, mask, frame_idx):

        _, K, H, W = mask.shape

        (frame, mask), pad = myutils.pad_divide_by([frame, mask], 16, (frame.size()[2], frame.size()[3]))

        frame = frame.expand(K, -1, -1, -1)  # obj_n, 3, h, w
        mask = mask[0].unsqueeze(1).float()  # obj_n, 1, h, w
        mask_ones = torch.ones_like(mask)
        mask_inv = (mask_ones - mask).clamp(0, 1)

        r4, r1 = self.encoder_m(frame, mask, mask_inv)
        h, w = r4.size(2), r4.size(3)
        r4 = self.position(r4.reshape(K, 1024, -1), frame_idx).reshape(K, 1024, h, w)

        k4, v4, h, w = self.keyvalue_r4(r4)  # num_objects, 128 and 512, H/16 * W/16

        k4_list = [k4[i] for i in range(K)]
        v4_list = [v4[i] for i in range(K)]
        # print('k4_list:',k4_list[0].size())  # (512,H/16 * W/16)
        # print('v4_list:',v4_list[0].size())  # (128,H/16 * W/16)
        return k4_list, v4_list, h, w

    def segment(self, frame, fb_global, mb, info):

        obj_n = fb_global.obj_n

        # pad
        if not self.training:
            [frame], pad = myutils.pad_divide_by([frame], 16, (frame.size()[2], frame.size()[3]))
        # print('frame:',frame.size())  # frame_idx, 3 , H , W
        r4, r3, r2, r1 = self.encoder_q(frame)

        # from tensorboardX import SummaryWriter
        # from torchvision.utils import make_grid
        # writer = SummaryWriter(comment=TIMESTAMP)
        #
        # writer.add_image('r4',
        #                  make_grid(r4[0].detach().cpu().unsqueeze(dim=1), nrow=10, padding=1, normalize=True,
        #                            pad_value=1, scale_each=True, range=(0, 1)), 0)
        # writer.add_image('value_bg',
        #                  make_grid(r4[0].detach().cpu().unsqueeze(dim=1), nrow=10, padding=1, normalize=True,
        #                            pad_value=1, scale_each=True, range=(0, 1)), 0)
        #
        # writer.flush()
        # writer.close()

        bs, _, global_match_h, global_match_w = r4.shape
        _, _, local_match_h, local_match_w = r1.shape

        k4, v4, h, w = self.keyvalue_r4(r4)  # 1, dim, H/16, W/16

        # kk4 = k4.reshape(k4.size(0), k4.size(1), r4.size(2), r4.size(3))
        # vv4 = v4.reshape(v4.size(0), v4.size(1), r4.size(2), r4.size(3))
        # print('kk4:', kk4.size())
        # print('vv4:', vv4.size())
        # from tensorboardX import SummaryWriter
        # from torchvision.utils import make_grid
        # writer = SummaryWriter(comment=TIMESTAMP)
        #
        # writer.add_image('key_bg',
        #                  make_grid(kk4[0].detach().cpu().unsqueeze(dim=1), nrow=10, padding=1, normalize=True,
        #                            pad_value=1, scale_each=True, range=(0,1)), 0)
        # writer.add_image('key_fg_1',
        #                  make_grid(kk4[1].detach().cpu().unsqueeze(dim=1), nrow=10, padding=1, normalize=True,
        #                            pad_value=1), 0)
        # writer.add_image('key_fg_2',
        #                  make_grid(kk4[2].detach().cpu().unsqueeze(dim=1), nrow=10, padding=1, normalize=True,
        #                            pad_value=1), 0)

        # writer.add_image('value_bg',
        #                  make_grid(vv4[0].detach().cpu().unsqueeze(dim=1), nrow=10, padding=1, normalize=True,
        #                            pad_value=1, scale_each=True, range=(0,1)), 0)
        # writer.add_image('value_fg_1',
        #                  make_grid(vv4[1].detach().cpu().unsqueeze(dim=1), nrow=10, padding=1, normalize=True,
        #                            pad_value=1), 0)
        # writer.add_image('value_fg_2',
        #                  make_grid(vv4[2].detach().cpu().unsqueeze(dim=1), nrow=10, padding=1, normalize=True,
        #                            pad_value=1), 0)

        # writer.flush()
        # writer.close()


        res_global = self.matcher(fb_global, k4, v4, global_match_h, global_match_w, mb, info)
        res_global = res_global.reshape(bs * obj_n, v4.shape[1] * 2, global_match_h, global_match_w)
        # print('res_global:',res_global.size())    # frame_idx * obj_n, 1024, h ,w

        r3_size = r3.shape
        r2_size = r2.shape
        r3 = r3.unsqueeze(1).expand(-1, obj_n, -1, -1, -1).reshape(bs * obj_n, *r3_size[1:])
        r2 = r2.unsqueeze(1).expand(-1, obj_n, -1, -1, -1).reshape(bs * obj_n, *r2_size[1:])
        r1_size = r1.shape
        r1 = r1.unsqueeze(1).expand(-1, obj_n, -1, -1, -1).reshape(bs * obj_n, *r1_size[1:])
        feature_size = (bs, obj_n, r1_size[2], r1_size[3])

        score = self.decoder(res_global, r3, r2, r1, feature_size)

        score = score.view(bs, obj_n, *frame.shape[-2:])  # frame_idx , obj_n , H , W

        if self.training:
            uncertainty = myutils.calc_uncertainty(F.softmax(score, dim=1))
            uncertainty = uncertainty.view(bs, -1).norm(p=2, dim=1) / math.sqrt(frame.shape[-2] * frame.shape[-1])
            uncertainty = uncertainty.mean()
        else:
            uncertainty = None

        score = torch.clamp(score, 1e-7, 1 - 1e-7)
        score = torch.log((score / (1 - score)))

        if not self.training:
            if pad[2] + pad[3] > 0:
                score = score[:, :, pad[2]:-pad[3], :]
            if pad[0] + pad[1] > 0:
                score = score[:, :, :, pad[0]:-pad[1]]

        return score, uncertainty

    def forward(self, *args, **kwargs):
        pass
