

import os
import torch
import torch.nn as nn
from torch import Tensor
import numpy as np
import math
import torch.nn.init as int
from UDL.Basis.variance_sacling_initializer import variance_scaling_initializer
from UDL.pansharpening.models.MISR.srcnn import SRCNN
from UDL.pansharpening.models.MISR.misr_public_modules import PixelShuffleBlock, DoubleConv2d
from kornia.geometry.transform import Resize
from UDL.Basis.postprocess import showimage8, showimage_grid8
import matplotlib.pyplot as plt
from UDL.pansharpening.models.MISR.generator import MultiTemporalGenerator
from UDL.pansharpening.models.MISR.gener_new import PhysMTMSynth
import torchvision.utils as vutils


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

class SRCNN_FusionNet(nn.Module):
    def __init__(self, spectral_num, criterion, channel=32):
        super(SRCNN_FusionNet, self).__init__()

        self.spectral_num = spectral_num
        self.criterion = criterion
        self._saved_train_images_done = False
        self.count = 0

        # ====== PanSharpening 分支 ====== #
        self.conv1 = nn.Conv2d(in_channels=spectral_num, out_channels=channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.res1 = Resblock()
        self.res2 = Resblock()
        self.res3 = Resblock()
        self.res4 = Resblock()
        self.conv3 = nn.Conv2d(in_channels=channel, out_channels=spectral_num, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.pansharpening_backbone = nn.Sequential(
            self.conv1,
            self.relu,
            self.res1,
            self.res2,
            self.res3,
            self.res4,
            self.conv3
        )

        # ====== MISR 分支  ====== #
        # 超参数设置
        self.in_channels =  spectral_num
        self.out_channels = spectral_num
        self.revisits = 8
        self.output_size = (64, 64)
        self.sr_kernel_size = 1
        self.zoom_factor = 4
        self.hidden_channels = 128  # 需要修改，根据zoom_factor而改变,保证hidden_channels % zoom_factor ** 2 == 0
        self.kernel_size = 3
        self.use_batchnorm = False
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
        self.encoder = DoubleConv2d(
            in_channels=self.in_channels,
            out_channels=self.hidden_channels,
            kernel_size=self.kernel_size,
            use_batchnorm=self.use_batchnorm,
        )
        self.fusion_SRCNN = SRCNN(
            residual_layers=1,
            hidden_channels=self.hidden_channels,
            kernel_size=self.kernel_size,
            revisits=self.revisits,
        )
        self.sr = PixelShuffleBlock(
            in_channels=self.hidden_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            sr_kernel_size=self.sr_kernel_size,
            zoom_factor=self.zoom_factor,
            use_batchnorm=self.use_batchnorm,
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
        self.misr_backbone = nn.Sequential(
            self.fusion_SRCNN,
            self.sr,
        )

        # ====== 汇总分支  ====== #
        self.color_correct = ColorCorrection(in_channels = spectral_num)

    def forward(self, x, y):  # x= lms; y = pan

        # ================= 进行MISR部分 ================= #
        # """
        #     输入为：x:[B,R,C,h,w], y:[B,c=1,H,W]
        #     输出为：[B,C,H,W]
        # """
        batch_size, revisits, channels, height, width = x.shape
        hidden_channels = self.hidden_channels
        x = x.view(batch_size * revisits, channels, height, width)
        x = self.encoder(x)
        x = x.view(batch_size, revisits * hidden_channels, height, width)
        x = self.misr_backbone(x)

        if self.training:
            x = self.resize_train(x)
        else :
            if self._saved_train_images_done:
                x = self.resize_train(x)
            else :
                x = self.resize_val(x)

        # ================= 进行PanSharpening部分 ================= #
        # """
        #     输入为：MISR的output:[B,C,H,W], y:[B,c=1,H,W]
        #     输出为：[B,C,H,W]
        # """
        pan_concat = y.repeat(1, self.spectral_num, 1, 1)  # Bsx8x64x64
        input = torch.sub(pan_concat, x)  # Bsx8x64x64
        output_pansharpening = self.pansharpening_backbone(input)

        # ================= 进行汇总部分 ================= #
        # """
        #     输入为：MISR的output:[B,C,H,W], PanSharpening的output:[B,C,H,W]
        #     输出为：[B,C,H,W]
        # """
        output = x + output_pansharpening
        output = self.color_correct(output)
        return output

    def train_step(self, data, *args, **kwargs):
        log_vars = {}
        gt, lms, ms, pan = data['gt'].cuda(), data['lms'].cuda(), \
                           data['ms'].cuda(), data['pan'].cuda()

        # 这里需要更改输入图像，当前ms为[B,C,h,w]、pan为[B,c=1,H,W]，需要保证有多时序图像输入，输入ms为[B,R,C,h,w]、pan为[B,c=1,H,W]
        '''
            # 暂时使用复制的方法，模拟构造revisited
            # ms_made =  ms.unsqueeze(1)
            # ms_made = ms_made.repeat(1, self.revisits, 1, 1, 1)
        '''
        # ms_made = ms.unsqueeze(1)
        # ms_made = self.generator_train.generate(ms_made)
        ms_made = self.gener.generate(gt)
        sr = self(ms_made, pan)

        # ================= 保存训练的部分生成的多时相多光谱的图像，到位置 --> save_name_output，用于观察 ================= #
        #     1. 调整格式： showimage8的输入格式是[h,w,C]，当前多时相多光谱图像 ms_made 的格式是[B,R,C,h,w]，所以需要每一批每一个revisited单独保存
        #     2. 保存图像个数： 总共只需要记录 5 对多时相多光谱图像 ms_made 即可，即可以选取第一批次的前 5 对多时相多光谱图像记录
        #     3. 每一对多时相多光谱图像之中有8张图像（即，revisited=8），可以利用showimage8提取图像之后，将这八张图像进行合并，保存为一张图片
        save_train_dir = r"/home/tongyufei/Pansharpen-DL-toolbox/UDL/results/image/SRCNN_FusionNet/train"
        if self.training and not getattr(self, "_saved_train_images_done", False):
            os.makedirs(save_train_dir, exist_ok=True)  # 若不存在就创建
            with torch.no_grad():
                num_to_save = min(5, ms_made.size(0))  # 只保存前5对（或batch不足5对则全存）
                for i in range(num_to_save):
                    sample = ms_made[i].detach().cpu()  # [R, C, h, w]
                    vis = showimage_grid8(sample)
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

        # 这里需要更改输入图像，当前ms为[B,C,h,w]、pan为[B,c=1,H,W]，需要保证有多时序图像输入，输入ms为[B,R,C,h,w]、pan为[B,c=1,H,W]
        # ms_made = ms.unsqueeze(1)
        # ms_made = self.generator_val.generate(ms_made)
        if gt.shape[1] > 8:  # 如果第二个维度大于8
            ms_made = self.gener.generate(gt.permute(0, 3, 1, 2))
        else:
            ms_made = self.gener.generate(gt)
        sr = self(ms_made, pan)

        # ================= 保存验证的部分生成的多时相多光谱的图像，到位置 --> save_name_output，用于观察 ================= #
        #     1. 调整格式： showimage8的输入格式是[h,w,C]，当前多时相多光谱图像x的格式是[B,R,C,h,w]，所以需要每一批每一个revisited单独保存
        #     2. 保存图像个数： 总共只需要记录 5 对多时相多光谱图像 ms_made 即可，即可以选取第一批次的前 5 对多时相多光谱图像记录
        #     3. 每一对多时相多光谱图像之中有8张图像（即，revisited=8），可以利用showimage8提取图像之后，将这八张图像进行合并，保存为一张图片
        save_val_dir = r"/home/tongyufei/Pansharpen-DL-toolbox/UDL/results/image/SRCNN_FusionNet/test"
        if not self.training and not getattr(self, "_saved_train_images_done", False):
            os.makedirs(save_val_dir, exist_ok=True)  # 若不存在就创建
            with torch.no_grad():
                num_to_save = min(5, ms_made.size(0))  # 只保存前5对（或batch不足5对则全存）
                for i in range(num_to_save):
                    sample = ms_made[i].detach().cpu()  # [R, C, h, w]
                    vis = showimage_grid8(sample)
                    save_path = os.path.join(save_val_dir, f"test_input_ms{self.count}.png")
                    plt.imsave(save_path, vis, dpi=300)
                    self.count = self.count +1


        return sr, gt

