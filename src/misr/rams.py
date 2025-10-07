from typing_extensions import Self
from src.datasources import JIF_S2_MEAN, JIF_S2_STD, S2_ALL_12BANDS
import numpy as np
import torch
import torch.nn.functional as F
from .misr_public_modules import RFAB
from .misr_public_modules import RTAB
from math import log2
from kornia.geometry.transform import Resize
from torch import nn, Tensor
from typing import Tuple, Optional
from src.transforms import Shift, WarpPerspective
import math
from einops import rearrange, repeat
class RAMS(nn.Module):
    """
    RAMS model for multi-frame super-resolution.
    """
    def __init__(self, scale, filters, kernel_size, depth, r, N, out_channels):
        super(RAMS, self).__init__()
        self.scale = scale                  # ​​Upsampling Scale Factor​​
        self.filters = filters              # the number of kernels
        self.kernel_size = kernel_size      # kernel size
        self.depth = depth                  # input number 
        self.r = r                          # Compression Rate in Attention Mechanisms
        self.N = N                          # number of RFAB in main block
        self.out_channels = out_channels   

        # Low-level feature extraction
        self.begin_conv3d = nn.Conv3d(3, self.filters, self.kernel_size, padding=1)
        self.end_conv3d = nn.Conv3d(self.filters, self.filters, self.kernel_size, padding=1)
        self.middle_conv3d = nn.Sequential(
            nn.Conv3d(
                in_channels=filters,
                out_channels=filters,
                kernel_size=kernel_size,  
                padding=(0, 1, 1),  
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
            kernel_size=(2, kernel_size, kernel_size),  
            padding=(0, 1, 1), 
        )
        self.before_upsampling_Con2d =  nn.Conv2d(
            in_channels=depth*3,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same",
            bias=False,
            padding_mode="reflect",
        )

        self.RFAB = RFAB(filters, kernel_size, r)
        self.RTAB = RTAB(depth*3, kernel_size, r)



    def forward(self, x):
        batch_size,  channels, revisits, height, width = x.shape
        x_global_res = x

        # ​main branch​​ 
        x = self.begin_conv3d(x)      # (B, F, T, H, W)
        x_res = x
        for i in range(self.N):       # 2. N BFAB
            x = self.RFAB(x)
        x = self.end_conv3d(x)        # 3. (B, F, T, H, W)
        x= x + x_res
        num_iterations = math.floor(self.depth / (self.kernel_size - 1) - 1)                #  Calculate the number of iterations, where T decreases by 2 each time, resulting in a total reduction of 2 * num_iterations
        num_iterations = int(num_iterations)
        for i in range(num_iterations):
            x = self.RFAB(x)
            x = self.middle_conv3d(x)
        if x.size(2) == 2:                                                                  # When even, T decreases by 1; when odd, T decreases by 2. Output shape: (B, 1, hidden_channels, H, W)
            x = self.before_upsampling_Con3d_even(x)
        else:
            x = self.before_upsampling_Con3d(x)
        x = x.squeeze(dim=2)                                                                # Squeeze the temporal dimension T, resulting in shape (B, hidden_channels, H, W)

        # 全局残差网络, 输入数据(B, C, T, H, W)
        x_global_res = x_global_res.permute(0, 2, 1, 3, 4)                                  # For 2D convolution, adjust the input dimensions from (B, C, T, H, W) to (B, T, C, H, W)
        x_global_res = x_global_res.view(batch_size, revisits * channels, height, width)    # (B, T*C, H, W)
        x_global_res = self.RTAB(x_global_res)
        x_global_res = self.before_upsampling_Con2d(x_global_res)

        x = x + x_global_res                                                                # (B, hidden_channels, H, W)
        return x                                                                            
