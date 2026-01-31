import os

import torch
import torch.nn as nn
import math

from matplotlib import pyplot as plt

from UDL.Basis.postprocess import showimage8
from UDL.Basis.variance_sacling_initializer import variance_scaling_initializer
from kornia.geometry.transform import Resize

from UDL.pansharpening.models.MISR.generator import MultiTemporalGenerator
from UDL.pansharpening.models.MISR.gener_new import PhysMTMSynth
from UDL.pansharpening.models import PanSharpeningModel

# -------------Initialization----------------------------------------
def init_weights(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                print("nn.Conv2D is initialized by variance_scaling_initializer")
                variance_scaling_initializer(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

class DiCNN(nn.Module):
    def __init__(self, spectral_num, criterion, channel=64, reg=True):
        super(DiCNN, self).__init__()
        self.criterion = criterion
        self.reg = reg
        self._saved_train_images_done = False
        self.count = 0

        # ConvTranspose2d: output = (input - 1)*stride + outpading - 2*padding + kernelsize
        self.conv1 = nn.Conv2d(in_channels=spectral_num + 1, out_channels=channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=channel, out_channels=spectral_num, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

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

        self.apply(init_weights)

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

        input = torch.cat([x, y], 1)  # Bsx9x64x64

        rs = self.relu(self.conv1(input))
        rs = self.relu(self.conv2(rs))
        out = self.conv3(rs)
        output = x + out

        return output

    def train_step(self, data, *args, **kwargs):
        log_vars = {}
        gt, lms, ms, pan = data['gt'].cuda(), data['lms'].cuda(), \
                                data['ms'].cuda(), data['pan'].cuda()

        # 这里需要更改输入图像，当前ms为[B,C,h,w]、pan为[B,c=1,H,W]，需要先生成多时序图像输入ms_made为[B,R,C,h,w],之后随机选择一张图像为ms_made作为输入(格式为[B,C,h,w])
        # ms_made = ms.unsqueeze(1)
        # ms_made = self.generator_train.generate(ms_made)  #生成多时序图像 [B, R, C, h, w]
        ms_made = self.gener.generate(gt)

        B, R, C, h, w = ms_made.shape
        rand_idx = torch.randint(low=0, high=R, size=(B,), device=ms.device)  # 在 0 ~ R-1 之间随机选择索引
        ms_selected = ms_made[torch.arange(B), rand_idx, ...]  # [B, C, h, w]

        sr = self(ms_selected, pan)

        # ================= 保存训练的部分多时相多光谱的图像，到位置 --> save_name_output，用于观察 ================= #
        #     1. 调整格式： showimage8的输入格式是[h,w,C]，当前多光谱图像ms的格式是[B,C,h,w]
        #     2. 保存图像个数： 总共只需要记录 5 张多光谱图像 ms 即可，即可以选取第一批次的前 5 张多光谱图像记录
        save_train_dir = r"/home/tongyufei/Pansharpen-DL-toolbox/UDL/results/image/DiCNN/train"
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

        loss = self.criterion(sr, gt, *args, **kwargs)
        # return sr, loss
        log_vars.update(loss=loss['loss'])
        return {'loss': loss['loss'], 'log_vars': log_vars}

    def val_step(self, data, *args, **kwargs):

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

        sr = self(ms_selected, pan)

        # ================= 保存训练的部分多光谱的图像，到位置 --> save_name_output，用于观察 ================= #
        #     1. 调整格式： showimage8的输入格式是[h,w,C]，当前多光谱图像ms的格式是[B,C,h,w]
        #     2. 保存图像个数： 总共只需要记录 5 张多光谱图像 ms 即可，即可以选取第一批次的前 5 张多光谱图像记录
        save_val_dir = r"/home/tongyufei/Pansharpen-DL-toolbox/UDL/results/image/DiCNN/test"
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


if __name__ == '__main__':
    lms = torch.randn([1, 8, 64, 64])
    pan = torch.randn([1, 8, 64, 64])
    model = DiCNN(8, None)
    x = model(lms, pan)
    print(x.shape)