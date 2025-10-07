from typing_extensions import Self
from src.datasources import JIF_S2_MEAN, JIF_S2_STD, S2_ALL_12BANDS
import numpy as np
import torch
import torch.nn.functional as F
from .misr_public_modules import ResidualBlock
from .misr_public_modules import DoubleConv2d
from math import log2
from kornia.geometry.transform import Resize
from torch import nn, Tensor
from typing import Tuple, Optional
from src.transforms import Shift, WarpPerspective
import math
from einops import rearrange, repeat

class SRCNN(nn.Module):
    """ Super-resolution CNN.
    Uses no recursive function, revisits are treated as channels.
    """

    def __init__(
        self,
        hidden_channels,
        revisits,
        kernel_size,
        residual_layers,
        use_batchnorm=False,
        **kws,
    ) -> None:
        """ Initialize the Super-resolution CNN.

        Parameters
        ----------
        in_channels : int
            The number of input channels.
        mask_channels : int
            The number of mask channels.
        revisits : int
            The number of revisits.
        hidden_channels : int
            The number of hidden channels.
        out_channels : int
            The number of output channels.
        kernel_size : int
            The kernel size.
        residual_layers : int
            The number of residual layers.
        output_size : tuple of int
            The output size.
        zoom_factor : int
            The zoom factor.
        sr_kernel_size : int
            The kernel size of the SR convolution.
        registration_kind : str, optional
            The kind of registration.
        homography_fc_size : int
            The size of the fully connected layer for the homography.
        use_reference_frame : bool, optional
            Whether to use the reference frame, by default False.
        """
        super().__init__()
        self.doubleconv2d = DoubleConv2d(
            in_channels=hidden_channels * revisits,  # revisits as channels
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            use_batchnorm=use_batchnorm,
        )
        self.residualblocks = nn.Sequential(
            *(
                ResidualBlock(
                    in_channels=hidden_channels,
                    kernel_size=kernel_size,
                    use_batchnorm=use_batchnorm,
                )
                for _ in range(residual_layers)
            )
        )
        self.fusion_SRCNN = nn.Sequential(self.doubleconv2d, self.residualblocks)

    def forward(
        self, x: Tensor, y: Optional[Tensor] = None, pan: Optional[Tensor] = None, mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = self.fusion_SRCNN(x)
        return x