import os
import torch
import torch.nn as nn
import numpy as np
import math
from UDL.Basis.variance_sacling_initializer import variance_scaling_initializer
from UDL.Basis.postprocess import showimage8
import matplotlib.pyplot as plt
from kornia.geometry.transform import Resize

from UDL.pansharpening.models.MISR.generator import MultiTemporalGenerator
from UDL.pansharpening.models.MISR.gener_new import PhysMTMSynth

import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, y=None, **kwargs):
        if y is not None:
            return self.fn(x, y, **kwargs) + x
        else:
            return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, y=None, **kwargs):
        if y is not None:
            return self.fn(self.norm(x), self.norm(y), **kwargs)
        else:
            return self.fn(self.norm(x), **kwargs)


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        # self.temperature = nn.Parameter(torch.ones(1, 1, 1, 1))
        self.sa1 = nn.Linear(dim, inner_dim, bias=False)
        self.sa2 = nn.Linear(dim, inner_dim, bias=False)
        self.se1 = nn.Linear(dim, inner_dim, bias=False)
        self.se2 = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, y, mask=None):
        b, n, _, h = *x.shape, self.heads
        y1 = rearrange(self.sa1(y), 'b n (h d) -> b h n d', h=h)
        y2 = rearrange(self.sa2(y), 'b n (h d) -> b h n d', h=h)
        x1 = rearrange(self.se1(x), 'b n (h d) -> b h n d', h=h)
        x2 = rearrange(self.se2(x), 'b n (h d) -> b h n d', h=h)
        sacm = (y1 @ y2.transpose(-2, -1)) * self.scale
        secm = (x1.transpose(-2, -1) @ x2) * self.scale / (n/self.dim_head)  # b h d d
        sacm = sacm.softmax(dim=-1)
        secm = secm.softmax(dim=-1)
        out1 = torch.einsum('b h i j, b h j d -> b h i d', sacm, x1)
        out2 = torch.einsum('b h n i, b h i j -> b h n j', y1, secm)
        out1 = rearrange(out1, 'b h n d -> b n (h d)')
        out2 = rearrange(out2, 'b h n d -> b n (h d)')
        out = out1 * out2
        out = self.to_out(out)
        return out


class S2Block(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, depth=1, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, MLP(dim, hidden_dim=mlp_dim, dropout=dropout)))]))

    def forward(self, x, y, mask=None):
        H = x.shape[2]
        x = rearrange(x, 'B C H W -> B (H W) C', H=H)
        y = rearrange(y, 'B C H W -> B (H W) C', H=H)
        for attn, ff in self.layers:
            x = attn(x, y, mask=mask)
            x = ff(x)
        x = rearrange(x, 'B (H W) C -> B C H W', H=H)
        return x


def init_weights(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                # variance_scaling_initializer(m.weight)
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


class ResBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels, hidden_channels, 3, 1, 1)
        self.conv1 = nn.Conv2d(hidden_channels, out_channels, 3, 1, 1)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        rs1 = self.relu(self.conv0(x))
        rs1 = self.conv1(rs1)
        rs = torch.add(x, rs1)
        return rs


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.upsamle = nn.Sequential(
            nn.Conv2d(n_feat, n_feat*16, 3, 1, 1, bias=False),
            nn.PixelShuffle(4)
        )

    def forward(self, x):
        return self.upsamle(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=False):
        super().__init__()
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0),
                nn.LeakyReLU()
            )
        else:
            self.up0 = nn.Sequential(
                nn.ConvTranspose2d(in_channels, in_channels, 2, 2, 0),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels, out_channels, 1, 1, 0),
                nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=out_channels),
                nn.LeakyReLU()
            )
            self.up1 = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, 2, 2, 0),
                nn.LeakyReLU()
            )
        self.conv = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.relu = nn.LeakyReLU()

    def forward(self, x1, x2):
        x1 = self.up1(x1)
        x = x1 + x2
        return self.relu(self.conv(x))


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=out_channels),
            nn.LeakyReLU()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 2, 2, 0),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=out_channels),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.conv(x)


class U2Net(nn.Module):
    def __init__(self, spectral_num, criterion, dim, dim_head=16, se_ratio_mlp=0.5, se_ratio_rb=0.5):
        super(U2Net, self).__init__()

        self.spectral_num = spectral_num
        self.criterion = criterion
        self._saved_train_images_done = False
        self.count = 0

        ms_dim = spectral_num
        pan_dim = 1

        self.relu = nn.LeakyReLU()
        self.upsample = Upsample(ms_dim)
        self.raise_ms_dim = nn.Sequential(
            nn.Conv2d(ms_dim, dim, 3, 1, 1),
            nn.LeakyReLU()
        )
        self.raise_pan_dim = nn.Sequential(
            nn.Conv2d(pan_dim, dim, 3, 1, 1),
            nn.LeakyReLU()
        )
        self.to_hrms = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(dim, ms_dim, 3, 1, 1)
        )

        dim0 = dim
        dim1 = int(dim0 * 2)
        dim2 = int(dim1 * 2)
        dim3 = dim1
        dim4 = dim0

        # layer 0
        self.s2block0 = S2Block(dim0, dim0 // dim_head, dim_head, int(dim0 * se_ratio_mlp))
        self.down0 = Down(dim0, dim1)
        self.resblock0 = ResBlock(dim0, int(se_ratio_rb * dim0), dim0)

        # layer 1
        self.s2block1 = S2Block(dim1, dim1 // dim_head, dim_head, int(dim1 * se_ratio_mlp))
        self.down1 = Down(dim1, dim2)
        self.resblock1 = ResBlock(dim1, int(se_ratio_rb * dim1), dim1)

        # layer 2
        self.s2block2 = S2Block(dim2, dim2 // dim_head, dim_head, int(dim2 * se_ratio_mlp))
        self.up0 = Up(dim2, dim3)
        self.resblock2 = ResBlock(dim2, int(se_ratio_rb * dim2), dim2)

        # layer 3
        self.s2block3 = S2Block(dim3, dim3 // dim_head, dim_head, int(dim3 * se_ratio_mlp))
        self.up1 = Up(dim3, dim4)
        self.resblock3 = ResBlock(dim3, int(se_ratio_rb * dim3), dim3)

        # layer 4
        self.s2block4 = S2Block(dim4, dim4 // dim_head, dim_head, int(dim4 * se_ratio_mlp))

        # ====== MISR 分支  ====== #
        # 超参数设置
        self.revisits = 8
        self.temporal_noise = 0.15

        self.gener = PhysMTMSynth(
            n_frames=8,
            scale=4
        )

        self.generator_train = MultiTemporalGenerator(
            n_frames=self.revisits,
            max_translation=1,  # 1像素偏移
            temporal_noise=self.temporal_noise,  # 3%噪声 0.30
        )
        self.generator_val = MultiTemporalGenerator(
            n_frames=self.revisits,
            max_translation=2,  # 2像素偏移
            temporal_noise=self.temporal_noise,  # 3%噪声 0.30
        )
        self.resize_train = Resize(
                (64, 64),
                interpolation="bilinear",
                align_corners=False,
                antialias=True,
            )
        self.resize_val = Resize(
                (72, 72),
                interpolation="bilinear",
                align_corners=False,
                antialias=True,
            )

    def forward(self, x, y):

        # ================= 模型的主要部分 ================= #
        # """
        #     输入为：x:[B,C,h,w], y:[B,c=1,H,W]
        #     输出为：[B,C,H,W]
        # """

        if self.training:
            x = self.resize_train(x)
        else :
            if self._saved_train_images_done:
                x = self.resize_train(x)
            else :
                x = self.resize_val(x)

        skip_c0 = x

        x = self.raise_ms_dim(x)
        y = self.raise_pan_dim(y)

        # layer 0
        x = self.s2block0(x, y)  # 32 64 64
        skip_c10 = x  # 32 64 64
        x = self.down0(x)  # 64 32 32
        y = self.resblock0(y)  # 32 64 64
        skip_c11 = y  # 32 64 64
        y = self.down0(y)  # 64 32 32

        # layer 1
        x = self.s2block1(x, y)  # 64 32 32
        skip_c20 = x
        x = self.down1(x)  # 128 16 16
        y = self.resblock1(y)  # 64 32 32
        skip_c21 = y  # 64 32 32
        y = self.down1(y)  # 128 16 16

        # layer 2
        x = self.s2block2(x, y)  # 128 16 16
        x = self.up0(x, skip_c20)  # 64 32 32
        y = self.resblock2(y)  # 128 16 16
        y = self.up0(y, skip_c21)  # 64 32 32

        # layer 3
        x = self.s2block3(x, y)  # 64 32 32
        x = self.up1(x, skip_c10)  # 32 64 64
        y = self.resblock3(y)  # 64 32 32
        y = self.up1(y, skip_c11)  # 32 64 64

        # layer 4
        x = self.s2block4(x, y)  # 32 64 64
        output = self.to_hrms(x)  # 8 64 64

        return output + skip_c0                                                                                        # 返回sr、misr/resize_ms

    def train_step(self, data, *args, **kwargs):
        log_vars = {}
        gt, lms, ms, pan = data['gt'].cuda(), data['lms'].cuda(), \
                           data['ms'].cuda(), data['pan'].cuda()

        # 这里需要更改输入图像，当前ms为[B,C,h,w]、pan为[B,c=1,H,W]，需要先生成多时序图像输入ms_made为[B,R,C,h,w],之后随机选择一张图像为ms_made作为输入(格式为[B,C,h,w])
        # ms_made = ms.unsqueeze(1)
        # ms_made = self.generator_train.generate(ms_made)  #生成多时序图像 [B, R, C, h, w]
        ms_made = self.gener.generate(gt)

        B, R, C, h, w = ms_made.shape
        rand_idx = torch.randint(low=0, high=R, size=(B,), device=ms.device)         # 在 0 ~ R-1 之间随机选择索引
        ms_selected = ms_made[torch.arange(B), rand_idx, ...]  # [B, C, h, w]

        sr = self(ms_selected, pan)

        # ================= 保存训练的部分多时相多光谱的图像，到位置 --> save_name_output，用于观察 ================= #
        #     1. 调整格式： showimage8的输入格式是[h,w,C]，当前多光谱图像ms的格式是[B,C,h,w]
        #     2. 保存图像个数： 总共只需要记录 5 张多光谱图像 ms 即可，即可以选取第一批次的前 5 张多光谱图像记录
        save_train_dir = r"/home/tongyufei/Pansharpen-DL-toolbox/UDL/results/image/U2Net/train"
        if self.training and not getattr(self, "_saved_train_images_done", False):
            os.makedirs(save_train_dir, exist_ok=True)  # 若不存在就创建
            with torch.no_grad():
                num_to_save = min(5, ms_selected.size(0))  # 只保存前5张（或batch不足5张则全存）
                for i in range(num_to_save):
                    # x: [B,C,H,W] -> [H,W,C]
                    img_hwc = ms_selected[i].detach().cpu().permute(1, 2, 0)
                    vis = showimage8(img_hwc)  # showimage8 期望 [H,W,C]
                    save_path = os.path.join(save_train_dir, f"train_input_ms{i + 1}.png")
                    plt.imsave(save_path, vis, dpi=300)
            self._saved_train_images_done = True  # 打一次性开关：以后不再保存

        loss = self.criterion(sr, gt, *args, **kwargs)['loss']
        log_vars.update(pan2ms=loss.item(), loss=loss.item())
        metrics = {'loss': loss, 'log_vars': log_vars}
        return metrics

    def val_step(self, data, *args, **kwargs):
        # gt, lms, ms, pan = data
        gt, lms, ms, pan = data['gt'].cuda(), data['lms'].cuda(), \
                           data['ms'].cuda(), data['pan'].cuda()

        # 这里需要更改输入图像，当前ms为[B,C,h,w]、pan为[B,c=1,H,W]，需要先生成多时序图像输入ms_made为[B,R,C,h,w],之后随机选择一张图像为ms_made作为输入(格式为[B,C,h,w])
        # ms_made = ms.unsqueeze(1)
        # ms_made = self.generator_val.generate(ms_made)  # 生成多时序图像 [B, R, C, h, w]
        if gt.shape[1] > 8:  # 如果第二个维度大于8
            ms_made = self.gener.generate(gt.permute(0, 3, 1, 2))
        else:
            ms_made = self.gener.generate(gt)

        B, R, C, h, w = ms_made.shape
        rand_idx = torch.randint(low=0, high=R, size=(B,), device=ms.device)  # 在 0 ~ R-1 之间随机选择索引
        ms_selected = ms_made[torch.arange(B), rand_idx, ...]  # [B, C, h, w]

        # sr = self(ms_selected, pan)

        #---------------------在推理时使用-------------------------------
        if not self.training and not getattr(self, "_saved_train_images_done", False):
            sr = self.tiled_inference(ms_selected, pan, cut_size=64, pad=4)
        else:
            sr = self(ms_selected, pan)

        # ================= 保存训练的部分多光谱的图像，到位置 --> save_name_output，用于观察 ================= #
        #     1. 调整格式： showimage8的输入格式是[h,w,C]，当前多光谱图像ms的格式是[B,C,h,w]
        #     2. 保存图像个数： 总共只需要记录 5 张多光谱图像 ms 即可，即可以选取第一批次的前 5 张多光谱图像记录
        save_val_dir = r"/home/tongyufei/Pansharpen-DL-toolbox/UDL/results/image/U2Net/test"
        if not self.training and not getattr(self, "_saved_train_images_done", False):
            os.makedirs(save_val_dir, exist_ok=True)  # 若不存在就创建
            with torch.no_grad():
                num_to_save = min(5, ms_selected.size(0))  # 只保存前5张（或batch不足5张则全存）
                for i in range(num_to_save):
                    # x: [B,C,H,W] -> [H,W,C]
                    img_hwc = ms_selected[i].detach().cpu().permute(1, 2, 0)
                    vis = showimage8(img_hwc)  # showimage8 期望 [H,W,C]
                    save_path = os.path.join(save_val_dir, f"test_input_ms{self.count + 1}.png")
                    plt.imsave(save_path, vis, dpi=300)
                    self.count = self.count + 1

        return sr, gt

    def tiled_inference(self, ms, pan, cut_size=64, pad=4):
        """
        ms: [B, C, h, w] (低分辨率)
        pan: [B, 1, H, W] (高分辨率)
        """
        B, C, h, w = ms.shape
        _, _, H, W = pan.shape

        ms_size = cut_size // 4  # 假设下采样倍率是4

        # 1. 计算需要补齐的边缘 (处理不能整除 cut_size 的情况)
        edge_H = (cut_size - (H % cut_size)) % cut_size
        edge_W = (cut_size - (W % cut_size)) % cut_size

        # 2. 对输入进行 Pad (反射填充以减少边界伪影)
        # ms 填充 pad//4, pan 填充 pad
        ms_pad = torch.nn.functional.pad(ms, (pad // 4, pad // 4, pad // 4, pad // 4), 'reflect')
        pan_pad = torch.nn.functional.pad(pan, (pad, pad, pad, pad), 'reflect')

        # 为了处理非整除情况，再次补齐到 cut_size 的倍数
        ms_pad = torch.nn.functional.pad(ms_pad, (0, edge_W // 4, 0, edge_H // 4), 'constant', 0)
        pan_pad = torch.nn.functional.pad(pan_pad, (0, edge_W, 0, edge_H), 'constant', 0)

        # 3. 准备输出容器
        output = torch.zeros(B, C, H + edge_H, W + edge_W).to(ms.device)

        scale_H = (H + edge_H) // cut_size
        scale_W = (W + edge_W) // cut_size

        # 4. 滑动窗口循环
        for i in range(scale_H):
            for j in range(scale_W):
                # 提取局部切块 (包含 pad)
                curr_ms = ms_pad[:, :, i * ms_size: (i + 1) * ms_size + pad // 2,
                          j * ms_size: (j + 1) * ms_size + pad // 2]
                curr_pan = pan_pad[:, :, i * cut_size: (i + 1) * cut_size + 2 * pad,
                           j * cut_size: (j + 1) * cut_size + 2 * pad]

                # 模型推理
                with torch.no_grad():
                    sr_tile = self.forward(curr_ms, curr_pan)  # 直接调 forward

                # 裁掉 pad 部分，填入输出容器
                output[:, :, i * cut_size: (i + 1) * cut_size,
                j * cut_size: (j + 1) * cut_size] = \
                    sr_tile[:, :, pad: cut_size + pad, pad: cut_size + pad]

        # 最后裁回原始尺寸
        return output[:, :, :H, :W]