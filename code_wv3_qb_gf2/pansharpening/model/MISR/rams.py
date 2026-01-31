from typing_extensions import Self
import numpy as np
import torch
import torch.nn.functional as F
from .misr_public_modules import RFAB
from .misr_public_modules import RTAB
from math import log2
from kornia.geometry.transform import Resize
from torch import nn, Tensor
from typing import Tuple, Optional
from .transforms import Shift, WarpPerspective
import math
from einops import rearrange, repeat
class RAMS(nn.Module):
    """
    RAMS model for multi-frame super-resolution.
    """
    def __init__(self, scale, filters, kernel_size, depth, r, N, out_channels):
        super(RAMS, self).__init__()
        self.scale = scale                  # 上采样放大倍数
        self.filters = filters              # 使用卷积核的个数
        self.kernel_size = kernel_size      # 卷积核的大小
        self.depth = depth                  # 输入图像的个数T
        self.r = r                          # 注意力机制的压缩率
        self.N = N                          # 主体网络中：RFAB的数量
        self.out_channels = out_channels    # 输出的通道数目（128）

        # Low-level feature extraction
        self.begin_conv3d = nn.Conv3d(4, self.filters, self.kernel_size, padding=1)
        self.end_conv3d = nn.Conv3d(self.filters, self.filters, self.kernel_size, padding=1)
        self.middle_conv3d = nn.Sequential(
            nn.Conv3d(
                in_channels=filters,
                out_channels=filters,
                kernel_size=kernel_size,  # 卷积核大小为 3
                padding=(0, 1, 1),  # 在 T 维度上不填充，H 和 W 维度上填充 1
            ),
            nn.ReLU(),
        )
        self.before_upsampling_Con3d = nn.Conv3d(
            self.filters,
            self.out_channels,
            kernel_size=kernel_size,
            padding=(0,1,1),
        )
        self.before_upsampling_Con3d_even = nn.Conv3d(
            filters,
            self.out_channels,
            kernel_size=(2, kernel_size, kernel_size),  # 使用 kernel_size=2 的时间维度
            padding=(0, 1, 1),  # 只在空间维度上填充
        )
        self.before_upsampling_Con2d =  nn.Conv2d(
            in_channels=depth*4,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same",
            bias=False,
            padding_mode="reflect",
        )

        self.RFAB = RFAB(filters, kernel_size, r)
        self.RTAB = RTAB(depth*4, kernel_size, r)



    def forward(self, x):
        # 输入数据(B, C, T, H, W)
        batch_size,  channels, revisits, height, width = x.shape
        x_global_res = x

        # 主分支
        x = self.begin_conv3d(x)      # 1. 初始的3D卷积，输出(B, F, T, H, W)
        x_res = x
        for i in range(self.N):       # 2. N个BFAB
            x = self.RFAB(x)
        x = self.end_conv3d(x)        # 3. 结尾的3D卷积，输出(B, F, T, H, W)
        x= x + x_res
        num_iterations = math.floor(self.depth / (self.kernel_size - 1) - 1)                #  计算循环次数，每次T减少2，一共减少2*num_iterations
        num_iterations = int(num_iterations)
        for i in range(num_iterations):
            x = self.RFAB(x)
            x = self.middle_conv3d(x)
        if x.size(2) == 2:                                                                  # 偶数时，T需要降低1，奇数时，T需要降低2；输出(B, 1, hidden_channels, H, W)
            x = self.before_upsampling_Con3d_even(x)
        else:
            x = self.before_upsampling_Con3d(x)
        x = x.squeeze(dim=2)                                                                # 去掉时间维度 T，形状变为 (B, hidden_channels, H, W)

        # 全局残差网络, 输入数据(B, C, T, H, W)
        x_global_res = x_global_res.permute(0, 2, 1, 3, 4)                                  # 2D卷积，需要调整输入，输入数据是 (B, C, T, H, W)，调整维度顺序为 (B, T, C, H, W)
        x_global_res = x_global_res.view(batch_size, revisits * channels, height, width)    # (B, T*C, H, W)
        x_global_res = self.RTAB(x_global_res)
        x_global_res = self.before_upsampling_Con2d(x_global_res)

        x = x + x_global_res                                                                # (B, hidden_channels, H, W)
        return x                                                                            # 输出数据是torch.Size([2, 128, 50, 50])
