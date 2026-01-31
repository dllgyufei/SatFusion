import numpy as np
import torch
from torch import nn, Tensor
import math
import torch.nn.functional as F
from kornia.geometry.transform import Resize

class HighFrequencyExtractor(nn.Module):
    def __init__(self, kernel_size=15):
        super().__init__()
        self.kernel_size = kernel_size
        # 使用均值滤波作为低通滤波
        self.blur = nn.AvgPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size//2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入张量 [B, C, H, W]
        Returns:
            高频分量 [B, C, H, W], 计算公式: x - blur(x)
        """
        x_blur = self.blur(x)
        hf = x - x_blur  # 高频 = 原始图像 - 模糊图像
        return hf


class ResidualBlock(nn.Module):
    """ResNet残差块，包含两个卷积层和跳跃连接"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU()

        # 如果输入输出通道数不一致，用1x1卷积调整维度
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        x += residual  # 残差连接
        x = self.relu(x)
        return x

class PANNet_With_MISR(nn.Module):
    """
    PANNet for Sharpening Task (Multi-spectral + Panchromatic fusion)
    Modified to match WorldStrat's input/output requirements
    """

    def __init__(self, in_channels=4, n1=64, n2=32, f1=9, f2=5, f3=5):
        super().__init__()

        # 高频提取器
        self.hf_extractor = HighFrequencyExtractor(kernel_size=15)

        # 初始卷积层（保持与原始PCNN一致）
        self.conv1 = nn.Conv2d(in_channels , n1, kernel_size=f1, padding=f1 // 2)

        # ResNet残差块（替换原始的第二层）
        self.res_block1 = ResidualBlock(n1, n1, kernel_size=f2)
        self.res_block2 = ResidualBlock(n1, n2, kernel_size=f2)

        # 输出层（保持与原始PCNN一致）
        self.conv3 = nn.Conv2d(n2, in_channels - 1, kernel_size=f3, padding=f3 // 2)


    def forward(self, ms: Tensor, pan: Tensor) -> Tensor:
        """
        Args:
            ms (Tensor): [B, C, H, W] multispectral images (after MISR)
            pan (Tensor): [B, 1, H, W] panchromatic image
        Returns:
            Tensor: [B, C, H, W] sharpened output
        """
        #ms_hf = self.hf_extractor(ms)  # [B, C, H, W]
        #pan_hf = self.hf_extractor(pan)  # [B, 1, H, W]
        ms_hf = ms  # [B, C, H, W]
        pan_hf = pan  # [B, 1, H, W]
        x = torch.cat([ms_hf, pan_hf], dim=1)  # [B, C+1, H, W]

        x = F.relu(self.conv1(x))  # 初始卷积
        x = self.res_block1(x)  # 残差块1
        x = self.res_block2(x)  # 残差块2
        x = self.conv3(x)  # 输出层（无激活函数）

        return x   # [B, C, H, W]

class PANNet_Only_Sharpening(nn.Module):
    """
    PANNet for Sharpening Task (Multi-spectral + Panchromatic fusion)
    Modified to match WorldStrat's input/output requirements
    """

    def __init__(self, in_channels=4, n1=64, n2=32, f1=9, f2=5, f3=5, output_size = (156,156)):
        super().__init__()
        self.output_size = output_size

        # 高频提取器
        self.hf_extractor = HighFrequencyExtractor(kernel_size=15)

        self.resize = Resize(
            self.output_size,
            interpolation="bilinear",
            align_corners=False,
            antialias=True,
        )

        # 初始卷积层（保持与原始PCNN一致）
        self.conv1 = nn.Conv2d(in_channels, n1, kernel_size=f1, padding=f1 // 2)

        # ResNet残差块（替换原始的第二层）
        self.res_block1 = ResidualBlock(n1, n1, kernel_size=f2)
        self.res_block2 = ResidualBlock(n1, n2, kernel_size=f2)

        # 输出层（保持与原始PCNN一致）
        self.conv3 = nn.Conv2d(n2, in_channels - 1, kernel_size=f3, padding=f3 // 2)

    def forward(self, ms: Tensor, pan: Tensor) -> Tensor:
        """
        Args:
            ms (Tensor): [B, C, h, w] multispectral images (without MISR)
            pan (Tensor): [B, 1, H, W] panchromatic image
        Returns:
            Tensor: [B, C, H, W] sharpened output
        """
        ms_res = self.resize(ms)
        #ms_hf = self.hf_extractor(ms)  # [B, C, h, w]
        #pan_hf = self.hf_extractor(pan)  # [B, 1, H, W]
        ms_hf = ms  # [B, C, H, W]
        pan_hf = pan  # [B, 1, H, W]
        ms_hf_up = self.resize(ms_hf)
        x = torch.cat([ms_hf_up, pan_hf], dim=1)

        x = F.relu(self.conv1(x))  # 初始卷积
        x = self.res_block1(x)  # 残差块1
        x = self.res_block2(x)  # 残差块2
        x = self.conv3(x)  # 输出层（无激活函数）

        return x + ms_res  # [B, C, H, W]

'''
class ResidualBlock(nn.Module):
    """残差块 (与原始实现一致)"""
    def __init__(self, channels, ksize=3, stride=1, pad=1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=ksize, stride=stride, padding=pad)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=ksize, stride=stride, padding=pad)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = out + identity  # 残差连接
        return out

class PANNet_With_MISR(nn.Module):

    def __init__(self, in_channels=3, num_res_blocks=4, output_size=(156, 156)):
        """
        Args:
            ms_channels: 多光谱波段数 (如 WorldStrat 的 4 波段)
            out_channels: 输出波段数 (通常为 3-RGB)
            num_res_blocks: 残差块数量 (默认 4)
        """
        super().__init__()
        # 高频提取器
        self.hf_extractor = HighFrequencyExtractor(kernel_size=5)
        self.first_conv = nn.Conv2d(in_channels + 1, 32, kernel_size=3, stride=1, padding=1)  # PAN+MS高频输入
        self.res_blocks = nn.Sequential(*[ResidualBlock(32) for _ in range(num_res_blocks)])
        self.last_conv = nn.Conv2d(32, in_channels, kernel_size=3, stride=1, padding=1)
        self.output_size = output_size

        self.resize = Resize(
            self.output_size,
            interpolation="bilinear",
            align_corners=False,
            antialias=True,
        )

    def forward(self, ms: Tensor, pan: Tensor) -> Tensor:
        """
        Args:
            ms (Tensor): [B, C, H, W] multispectral images (with MISR)
            pan (Tensor): [B, 1, H, W] panchromatic image
        Returns:
            Tensor: [B, C, H, W] sharpened output
        """
        ms_hf = self.hf_extractor(ms)  # [B, C, h, w]
        pan_hf = self.hf_extractor(pan)  # [B, 1, H, W]

        # 拼接 PAN 和 MS 高频特征
        x = torch.cat([pan_hf, ms_hf], dim=1)  # [B, ms_channels+1, H, W]

        # 特征融合
        x = F.relu(self.first_conv(x))
        x = self.res_blocks(x)

        # 细节注入 (限制输出通道数与 up_ms 一致)
        out = self.last_conv(x) 
        return out

class PANNet_Only_Sharpening(nn.Module):

    def __init__(self, in_channels=3, num_res_blocks=4, output_size=(156,156)):
        """
        Args:
            ms_channels: 多光谱波段数 (如 WorldStrat 的 4 波段)
            out_channels: 输出波段数 (通常为 3-RGB)
            num_res_blocks: 残差块数量 (默认 4)
        """
        super().__init__()
        # 高频提取器
        self.hf_extractor = HighFrequencyExtractor(kernel_size=5)
        self.first_conv = nn.Conv2d(in_channels + 1, 32, kernel_size=3, stride=1, padding=1)  # PAN+MS高频输入
        self.res_blocks = nn.Sequential(*[ResidualBlock(32) for _ in range(num_res_blocks)])
        self.last_conv = nn.Conv2d(32, in_channels, kernel_size=3, stride=1, padding=1)
        self.output_size = output_size
        self.resize = Resize(
            self.output_size,
            interpolation="bilinear",
            align_corners=False,
            antialias=True,
        )

    def forward(self,  ms: Tensor, pan: Tensor) -> Tensor:
        """
        Args:
            ms (Tensor): [B, C, h, w] multispectral images (without MISR)
            pan (Tensor): [B, 1, H, W] panchromatic image
        Returns:
            Tensor: [B, C, H, W] sharpened output
        """
        ms_res = self.resize(ms)
        ms_hf = self.hf_extractor(ms)  # [B, C, h, w]
        pan_hf = self.hf_extractor(pan)  # [B, 1, H, W]

        # 拼接 PAN 和 MS 高频特征
        x = torch.cat([pan_hf, ms_hf], dim=1)  # [B, ms_channels+1, H, W]

        # 特征融合
        x = F.relu(self.first_conv(x))
        x = self.res_blocks(x)

        # 细节注入 (限制输出通道数与 up_ms 一致)
        out = self.last_conv(x) + ms_res
        return out
'''