import os
import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.init as int

import torch
import torch.nn as nn
from torch import Tensor
import matplotlib.pyplot as plt
import math
from UDL.Basis.variance_sacling_initializer import variance_scaling_initializer
from UDL.pansharpening.models import PanSharpeningModel
from UDL.Basis.postprocess import showimage8, showimage_grid8
from kornia.geometry.transform import Resize
from UDL.pansharpening.models.MISR.srcnn import SRCNN
from UDL.pansharpening.models.MISR.misr_public_modules import PixelShuffleBlock, DoubleConv2d
from UDL.pansharpening.models.MISR.generator import MultiTemporalGenerator
from UDL.pansharpening.models.MISR.gener_new import PhysMTMSynth

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

class SRCNN_MSDCNN(nn.Module):
    def __init__(self, spectral_num, criterion, channel=64):
        super(SRCNN_MSDCNN, self).__init__()
        self._saved_train_images_done = False
        self.count = 0

        self.criterion = criterion

        input_channel = spectral_num + 1
        output_channel = spectral_num

        self.conv1 = nn.Conv2d(in_channels=input_channel, out_channels=60, kernel_size=7, stride=1, padding=3, bias=True)

        self.conv2_1 = nn.Conv2d(in_channels=60, out_channels=20, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2_2 = nn.Conv2d(in_channels=60, out_channels=20, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv2_3 = nn.Conv2d(in_channels=60, out_channels=20, kernel_size=7, stride=1, padding=3, bias=True)

        self.conv3 = nn.Conv2d(in_channels=60, out_channels=30, kernel_size=3, stride=1, padding=1, bias=True)

        self.conv4_1 = nn.Conv2d(in_channels=30, out_channels=10, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv4_2 = nn.Conv2d(in_channels=30, out_channels=10, kernel_size=5, stride=1, padding=2, bias=True)
        self.conv4_3 = nn.Conv2d(in_channels=30, out_channels=10, kernel_size=7, stride=1, padding=3, bias=True)

        self.conv5 = nn.Conv2d(in_channels=30, out_channels=output_channel, kernel_size=5, stride=1, padding=2, bias=True)

        self.shallow1 = nn.Conv2d(in_channels=input_channel, out_channels=64, kernel_size=9, stride=1, padding=4, bias=True)
        self.shallow2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0, bias=True)
        self.shallow3 = nn.Conv2d(in_channels=32, out_channels=output_channel, kernel_size=5, stride=1, padding=2, bias=True)

        self.relu = nn.ReLU(inplace=True)

        # ====== MISR 分支  ====== #
        # 超参数设置
        self.in_channels =   spectral_num
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


    def forward(self, x, y):

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
        else:
            if self._saved_train_images_done:
                x = self.resize_train(x)
            else:
                x = self.resize_val(x)

        # ================= 进行PanSharpening部分 ================= #
        # """
        #     输入为：MISR的output:[B,C,H,W], y:[B,c=1,H,W]
        #     输出为：[B,C,H,W]
        # """
        concat = torch.cat([x, y], 1)  # Bsx9x64x64

        out1 = self.relu(self.conv1(concat))  # Bsx60x64x64
        out21 = self.conv2_1(out1)   # Bsx20x64x64
        out22 = self.conv2_2(out1)   # Bsx20x64x64
        out23 = self.conv2_3(out1)   # Bsx20x64x64
        out2 = torch.cat([out21, out22, out23], 1)  # Bsx60x64x64

        out2 = self.relu(torch.add(out2, out1))  # Bsx60x64x64

        out3 = self.relu(self.conv3(out2))  # Bsx30x64x64
        out41 = self.conv4_1(out3)          # Bsx10x64x64
        out42 = self.conv4_2(out3)          # Bsx10x64x64
        out43 = self.conv4_3(out3)          # Bsx10x64x64
        out4 = torch.cat([out41, out42, out43], 1)  # Bsx30x64x64

        out4 = self.relu(torch.add(out4, out3))  # Bsx30x64x64

        out5 = self.conv5(out4)  # Bsx8x64x64

        shallow1 = self.relu(self.shallow1(concat))   # Bsx64x64x64
        shallow2 = self.relu(self.shallow2(shallow1))  # Bsx32x64x64
        shallow3 = self.shallow3(shallow2) # Bsx8x64x64

        out = torch.add(out5, shallow3)  # Bsx8x64x64
        out = self.relu(out)  # Bsx8x64x64

        # ================= 进行汇总部分 ================= #
        # """
        #     输入为：MISR的output:[B,C,H,W], PanSharpening的output:[B,C,H,W]
        #     输出为：[B,C,H,W]
        # """
        output = x + out
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
        save_train_dir = r"/home/tongyufei/Pansharpen-DL-toolbox/UDL/results/image/SRCNN_MSDCNN/train"
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

        loss = self.criterion(sr, gt, *args, **kwargs)
        # return sr, loss
        log_vars.update(loss=loss['loss'])
        return {'loss': loss['loss'], 'log_vars': log_vars}

    def val_step(self, data, *args, **kwargs):

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
        save_val_dir = r"/home/tongyufei/Pansharpen-DL-toolbox/UDL/results/image/SRCNN_MSDCNN/test"
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



if __name__ == '__main__':
    lms = torch.randn([1, 8, 64, 64])
    pan = torch.randn([1, 1, 64, 64])
    ms = torch.randn([1, 8, 16, 16])
    # model = BDPN(8, None)
    # x,_ = model(ms, pan)
    # print(x.shape)