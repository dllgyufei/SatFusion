import numpy as np
import torch
from torch import nn, Tensor
import math
import torch.nn.functional as F
from kornia.geometry.transform import Resize

class ResidualBlock(nn.Module):
    """ResNet Residual Block with Two Convolutional Layers and Skip Connectionn"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU()

        # adjust the dimension
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        x += residual  # residual connection 
        x = self.relu(x)
        return x

class PANNet_With_MISR(nn.Module):
    """
    PANNet for Sharpening Task (Multi-spectral + Panchromatic fusion)
    Modified to match WorldStrat's input/output requirements
    """

    def __init__(self, in_channels=4, n1=64, n2=32, f1=9, f2=5, f3=5):
        super().__init__()

        # Initial Convolution Block (the origin high-frequency extractor is inappropriate for small-sized images)​
        self.conv1 = nn.Conv2d(in_channels , n1, kernel_size=f1, padding=f1 // 2)

        # ResNet Residual Block (Replacing the Original Second Layer)​
        self.res_block1 = ResidualBlock(n1, n1, kernel_size=f2)
        self.res_block2 = ResidualBlock(n1, n2, kernel_size=f2)

        # ​​Output Layer (Consistent with Original PCNN)​
        self.conv3 = nn.Conv2d(n2, in_channels - 1, kernel_size=f3, padding=f3 // 2)


    def forward(self, ms: Tensor, pan: Tensor) -> Tensor:
        """
        Args:
            ms (Tensor): [B, C, H, W] multispectral images (after MISR)
            pan (Tensor): [B, 1, H, W] panchromatic image
        Returns:
            Tensor: [B, C, H, W] sharpened output
        """
        ms_hf = ms  # [B, C, H, W]
        pan_hf = pan  # [B, 1, H, W]
        x = torch.cat([ms_hf, pan_hf], dim=1)  # [B, C+1, H, W]

        x = F.relu(self.conv1(x))  # Initial Convolution Block
        x = self.res_block1(x)  # ResNet Residual Block1
        x = self.res_block2(x)  # ResNet Residual Block2
        x = self.conv3(x)  # ​​Output Layer (without activation function​​)

        return x   # [B, C, H, W]

class PANNet_Only_Sharpening(nn.Module):
    """
    PANNet for Sharpening Task (Multi-spectral + Panchromatic fusion)
    Modified to match WorldStrat's input/output requirements
    """

    def __init__(self, in_channels=4, n1=64, n2=32, f1=9, f2=5, f3=5, output_size = (156,156)):
        super().__init__()
        self.output_size = output_size



        self.resize = Resize(
            self.output_size,
            interpolation="bilinear",
            align_corners=False,
            antialias=True,
        )

        self.conv1 = nn.Conv2d(in_channels, n1, kernel_size=f1, padding=f1 // 2)

        self.res_block1 = ResidualBlock(n1, n1, kernel_size=f2)
        self.res_block2 = ResidualBlock(n1, n2, kernel_size=f2)


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
        ms_hf = ms  # [B, C, H, W]
        pan_hf = pan  # [B, 1, H, W]
        ms_hf_up = self.resize(ms_hf)
        x = torch.cat([ms_hf_up, pan_hf], dim=1)

        x = F.relu(self.conv1(x))  
        x = self.res_block1(x)  
        x = self.res_block2(x)  
        x = self.conv3(x)  

        return x + ms_res  # [B, C, H, W]

