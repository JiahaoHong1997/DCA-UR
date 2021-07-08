import math
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet18

import myutils
from model import TimeEncoding


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


class KeyProjection(nn.Module):
    def __init__(self, indim, keydim):
        super().__init__()
        self.key_proj = nn.Conv2d(indim, keydim, kernel_size=3, padding=1)

        nn.init.orthogonal_(self.key_proj.weight.data)
        nn.init.zeros_(self.key_proj.bias.data)

    def forward(self, x):
        return self.key_proj(x)

class FeatureFusionBlock(nn.Module):
    def __init__(self, indim, outdim):
        super().__init__()

        self.block1 = ResBlock(indim, outdim)
        self.attention = CBAM(outdim)
        self.block2 = ResBlock(outdim, outdim)

    def forward(self, x, f16):
        x = torch.cat([x, f16], 1)
        x = self.block1(x)
        r = self.attention(x)
        x = self.block2(x + r)

        return x


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

class KeyEncoder(nn.Module):
    def __init__(self, load_imagenet_params):
        super(KeyEncoder, self).__init__()

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

class ValueEncoder(nn.Module):
    def __init__(self, load_imagenet_params):
        super(ValueEncoder, self).__init__()

        self.conv1_m = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1_o = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        resnet = resnet18(load_imagenet_params)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu  # 1/2, 64
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # 1/4, 64
        self.layer2 = resnet.layer2  # 1/8, 128
        self.layer3 = resnet.layer3  # 1/16, 256

        self.fuser = FeatureFusionBlock(1024 + 256, 512)

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, in_f, r4, in_m, in_o):

        f = (in_f - self.mean) / self.std
        # key_f16 is the feature from the key encoder

        x = self.conv1(f) + self.conv1_m(in_m) + self.conv1_o(in_o)

        x = self.bn1(x)
        x = self.relu(x)   # 1/2, 64
        x = self.maxpool(x)  # 1/4, 64
        x = self.layer1(x)   # 1/4, 64
        x = self.layer2(x)  # 1/8, 128
        x = self.layer3(x)  # 1/16, 256

        x = self.fuser(x, r4)

        return x


class Matcher(nn.Module):
    def __init__(self):
        super(Matcher, self).__init__()

    def forward(self, feature_bank, q_in, q_out, mask_bank):
        mem_out_list = []

        for i in range(0, feature_bank.obj_n):
            d_key, bank_n = feature_bank.keys[i].size()  # 128 , t*h*w

            bs, _, n = q_in.size()
            p = torch.matmul(feature_bank.keys[i].transpose(0, 1), q_in) / math.sqrt(d_key)  # bs, t*h*w, h*w
            p = F.softmax(p, dim=1)   # bs, t*h*w, h*w

            mem = torch.matmul(feature_bank.values[i], p)  # frame_idx, 512, h*w
            mask_mem = torch.matmul(mask_bank.mask_list[i], p)  # 1, 1, h*w
            q_out_with_mask = q_out * mask_mem  # Location Guidance

            mem_out_list.append(torch.cat([mem, q_out_with_mask], dim=1))  # frame_idx, 1024, h*w

        mem_out_tensor = torch.stack(mem_out_list, dim=0).transpose(0, 1)  # frame_idx, obj_n, 1024, h*w

        return mem_out_tensor



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

        m4 = self.ResMM(self.convFM(r4))  # 1024 -> hidden_dim  1/16
        m3 = self.RF3(r3, m4)  # hidden_dim  1/8
        m2 = self.RF2(r2, m3)  # hidden_dim  1/4

        p2 = self.pred2(F.relu(m2))
        p = F.interpolate(p2, scale_factor=2, mode='bilinear', align_corners=False)  # 1

        bs, obj_n, h, w = feature_shape
        rough_seg = F.softmax(p, dim=1)[:, 1]
        rough_seg = rough_seg.view(bs, obj_n, h, w)
        rough_seg = F.softmax(rough_seg, dim=1)  # object-level normalization

        # Local refinement
        uncertainty = myutils.calc_uncertainty(rough_seg)
        uncertainty = uncertainty.expand(-1, obj_n, -1, -1).reshape(bs * obj_n, 1, h, w)

        rough_seg = rough_seg.view(bs * obj_n, 1, h, w)  # bs*obj_n, 1, h, w
        r1_weighted = r1 * rough_seg
        r1_local = self.local_max(r1_weighted)  # bs*obj_n, 64, h, w
        r1_local = r1_local / (self.local_max(rough_seg) + 1e-8)  # neighborhood reference
        r1_conf = self.local_avg(rough_seg)  # bs*obj_n, 1, h, w

        local_match = torch.cat([r1, r1_local], dim=1)
        q = self.local_ResMM(self.local_convFM(local_match))
        q = r1_conf * self.local_pred2(F.relu(q))

        p = p + uncertainty * q
        p = F.interpolate(p, scale_factor=2, mode='bilinear', align_corners=False)

        p = F.softmax(p, dim=1)[:, 1]

        return p


class TELG(nn.Module):
    def __init__(self, device, load_imagenet_params=False):
        super(TELG, self).__init__()

        self.key_encoder = KeyEncoder(load_imagenet_params)
        self.value_encoder = ValueEncoder(load_imagenet_params)

        # Projection from f16 feature space to key space
        self.key_proj = KeyProjection(1024, keydim=64)

        # Compress f16 a bit to use in ecoding later on
        self.key_comp = nn.Conv2d(1024, 512, kernel_size=3, padding=1)

        self.te = TimeEncoding.TimeEncoding(d_hid=1024)

        self.matcher = Matcher()

        self.decoder = Decoder(device)

    def memorize(self, frame, mask, frame_idx):

        _, K, H, W = mask.shape

        (frame, mask), pad = myutils.pad_divide_by([frame, mask], 16, (frame.size()[2], frame.size()[3]))

        frame = frame.expand(K, -1, -1, -1)
        mask = mask[0].unsqueeze(1).float()
        mask_ones = torch.ones_like(mask)
        mask_inv = (mask_ones - mask).clamp(0, 1)

        r4, _, _, _ = self.key_encoder(frame)
        h, w = r4.size(2), r4.size(3)
        r4 = self.te(r4.reshape(K, 1024, -1), frame_idx).reshape(K, 1024, h, w)
        k4 = self.key_proj(r4)
        v4 = self.value_encoder(frame, r4, mask, mask_inv)

        k4 = k4.reshape(K, 64, h*w)
        v4 = v4.reshape(K, 512, h*w)

        k4_list = [k4[i] for i in range(K)]
        v4_list = [v4[i] for i in range(K)]
        return k4_list, v4_list, h, w

    def segment(self, frame, fb_global, mb, info):

        obj_n = fb_global.obj_n

        # pad
        if not self.training:
            [frame], pad = myutils.pad_divide_by([frame], 16, (frame.size()[2], frame.size()[3]))

        r4, r3, r2, r1 = self.key_encoder(frame)
        bs, _, global_match_h, global_match_w = r4.shape
        _, _, local_match_h, local_match_w = r1.shape

        k4 = self.key_proj(r4)
        v4 = self.key_comp(r4)

        k4 = k4.view(-1, 64, global_match_h*global_match_w)
        v4 = v4.reshape(-1, 512, global_match_h*global_match_w)

        res_global = self.matcher(fb_global, k4, v4, mb)
        res_global = res_global.reshape(bs * obj_n, 1024, global_match_h, global_match_w)

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
