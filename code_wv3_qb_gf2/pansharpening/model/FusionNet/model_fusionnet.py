

import os
import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.init as int
from UDL.Basis.variance_sacling_initializer import variance_scaling_initializer
from UDL.Basis.postprocess import showimage8
import matplotlib.pyplot as plt
from kornia.geometry.transform import Resize

from UDL.pansharpening.models.MISR.generator import MultiTemporalGenerator
from UDL.pansharpening.models.MISR.gener_new import PhysMTMSynth


class loss_with_l2_regularization(nn.Module):
    def __init__(self):
        super(loss_with_l2_regularization, self).__init__()

    def forward(self, criterion, model, weight_decay=1e-5, flag=False):
        regularizations = []
        for k, v in model.named_parameters():
            if 'conv' in k and 'weight' in k:
                # print(k)
                penality = weight_decay * ((v.data ** 2).sum() / 2)
                regularizations.append(penality)
                if flag:
                    print("{} : {}".format(k, penality))
        # r = torch.sum(regularizations)

        loss = criterion + sum(regularizations)
        return loss


# -------------Initialization----------------------------------------
def init_weights(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):  ## initialization for Conv2d
                print("initial nn.Conv2d with var_scale_new: ", m)
                # try:
                #     import tensorflow as tf
                #     tensor = tf.get_variable(shape=m.weight.shape, initializer=tf.variance_scaling_initializer(seed=1))
                #     m.weight.data = tensor.eval()
                # except:
                #     print("try error, run variance_scaling_initializer")
                # variance_scaling_initializer(m.weight)
                variance_scaling_initializer(m.weight)  # method 1: initialization
                # nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')  # method 2: initialization
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):  ## initialization for BN
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):  ## initialization for nn.Linear
                # variance_scaling_initializer(m.weight)
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


# -------------ResNet Block (One)----------------------------------------
class Resblock(nn.Module):
    def __init__(self):
        super(Resblock, self).__init__()

        channel = 32
        self.conv20 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1,
                                bias=True)
        self.conv21 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1,
                                bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):  # x= hp of ms; y = hp of pan
        rs1 = self.relu(self.conv20(x))  # Bsx32x64x64
        rs1 = self.conv21(rs1)  # Bsx32x64x64
        rs = torch.add(x, rs1)  # Bsx32x64x64
        return rs

class ColorCorrection(nn.Module):
    def __init__(self, in_channels=4):
        super().__init__()
        # 1x1卷积网络（不改变H/W）
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),  # 通道扩展
            nn.ReLU(inplace=True),
            nn.Conv2d(64, in_channels, kernel_size=1)  # 恢复原始通道
        )

    def forward(self, x):
        """
        输入: x [B,C,H,W] (C通常为3-RGB)
        输出: [B,C,H,W] (H/W不变)
        """
        return x + self.conv(x)  # 残差连接：原始图像 + 学习的色彩偏移

class FusionNet(nn.Module):
    def __init__(self, spectral_num, criterion, channel=32):
        super(FusionNet, self).__init__()

        self.spectral_num = spectral_num
        self.criterion = criterion
        self._saved_train_images_done = False
        self.count = 0

        self.conv1 = nn.Conv2d(in_channels=spectral_num, out_channels=channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.res1 = Resblock()
        self.res2 = Resblock()
        self.res3 = Resblock()
        self.res4 = Resblock()
        self.conv3 = nn.Conv2d(in_channels=channel, out_channels=spectral_num, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.backbone = nn.Sequential(  # method 2: 4 resnet repeated blocks
            self.res1,
            self.res2,
            self.res3,
            self.res4
        )

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
                (256, 256),
                interpolation="bilinear",
                align_corners=False,
                antialias=True,
            )

        # # ====== 汇总分支  ====== #
        # self.color_correct = ColorCorrection(in_channels = spectral_num)

    def forward(self, x, y):  # x= lms; y = pan

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

        pan_concat = y.repeat(1, self.spectral_num, 1, 1)  # Bsx8x64x64
        input = torch.sub(pan_concat, x)  # Bsx8x64x64
        rs = self.relu(self.conv1(input))  # Bsx32x64x64
        rs = self.backbone(rs)  # ResNet's backbone!
        output = self.conv3(rs)  # Bsx8x64x64
        return output + x

        # output = x + output
        # output = self.color_correct(output)
        # return output

    def train_step(self, data, *args, **kwargs):
        log_vars = {}
        gt, lms, ms, pan = data['gt'].cuda(), data['lms'].cuda(), \
                           data['ms'].cuda(), data['pan'].cuda()

        # 这里需要更改输入图像，当前ms为[B,C,h,w]、pan为[B,c=1,H,W]，需要先生成多时序图像输入ms_made为[B,R,C,h,w],之后随机选择一张图像为ms_made作为输入(格式为[B,C,h,w])
        ms_made = self.gener.generate(gt)
        B, R, C, h, w = ms_made.shape
        rand_idx = torch.randint(low=0, high=R, size=(B,), device=ms.device)         # 在 0 ~ R-1 之间随机选择索引
        ms_selected = ms_made[torch.arange(B), rand_idx, ...]  # [B, C, h, w]

        sr = self(ms_selected, pan)
        # res = self(lms, pan)
        # sr = lms + res  # output:= lms + hp_sr

        # ================= 保存训练的部分多时相多光谱的图像，到位置 --> save_name_output，用于观察 ================= #
        #     1. 调整格式： showimage8的输入格式是[h,w,C]，当前多光谱图像ms的格式是[B,C,h,w]
        #     2. 保存图像个数： 总共只需要记录 5 张多光谱图像 ms 即可，即可以选取第一批次的前 5 张多光谱图像记录
        save_train_dir = r"/home/tongyufei/Pansharpen-DL-toolbox/UDL/results/image/FusionNet/train"
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

        #这里需要更改输入图像，当前ms为[B,C,h,w]、pan为[B,c=1,H,W]，需要先生成多时序图像输入ms_made为[B,R,C,h,w],之后随机选择一张图像为ms_made作为输入(格式为[B,C,h,w])
        if gt.shape[1] > 8:  # 如果第二个维度大于8
            ms_made = self.gener.generate(gt.permute(0, 3, 1, 2))
        else:
            ms_made = self.gener.generate(gt)
        B, R, C, h, w = ms_made.shape
        rand_idx = torch.randint(low=0, high=R, size=(B,), device=ms.device)  # 在 0 ~ R-1 之间随机选择索引
        ms_selected = ms_made[torch.arange(B), rand_idx, ...]  # [B, C, h, w]

        sr = self(ms_selected, pan)
        # sr = self(lms, pan)
        # sr = lms + res  # output:= lms + hp_sr

        # ================= 保存训练的部分多光谱的图像，到位置 --> save_name_output，用于观察 ================= #
        #     1. 调整格式： showimage8的输入格式是[h,w,C]，当前多光谱图像ms的格式是[B,C,h,w]
        #     2. 保存图像个数： 总共只需要记录 5 张多光谱图像 ms 即可，即可以选取第一批次的前 5 张多光谱图像记录
        save_val_dir = r"/home/tongyufei/Pansharpen-DL-toolbox/UDL/results/image/FusionNet/test"
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