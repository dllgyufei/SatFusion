from typing_extensions import Self
from src.datasources import JIF_S2_MEAN, JIF_S2_STD, S2_ALL_12BANDS
import numpy as np
import torch
import torch.nn.functional as F
from .misr_public_modules import FusionBlock
from math import log2
from kornia.geometry.transform import Resize
from torch import nn, Tensor
from typing import Tuple, Optional
from src.transforms import Shift, WarpPerspective
import math
from einops import rearrange, repeat
class RecursiveFusion(nn.Module):
    """ Recursively fuses a set of encodings. """

    def __init__(self, in_channels, kernel_size, revisits):
        """ Initialize the RecursiveFusion layer.

        Parameters
        ----------
        in_channels : int
            The number of input channels.
        kernel_size : int
            The kernel size.
        revisits : int
            The number of revisits.
        """
        super().__init__()

        log2_revisits = log2(revisits)
        if log2_revisits % 1 == 0:
            num_fusion_layers = int(log2_revisits)
        else:
            num_fusion_layers = int(log2_revisits) + 1

        pairwise_fusion = FusionBlock(in_channels, kernel_size, use_batchnorm=False)

        self.fusion = nn.Sequential(
            *(pairwise_fusion for _ in range(num_fusion_layers))
        )

    @staticmethod
    def pad(x):
        """ Pad the input tensor with black revisits to a power of 2.

        Parameters
        ----------
        x : Tensor
            The input tensor of shape (batch_size, revisits, in_channels, height, width).

        Returns
        -------
        Tensor
            The output tensor of shape (batch_size, revisits, in_channels, height, width).
        """

        # TODO: should we pad with copies of revisits instead of zeros?
        # TODO: move to transforms.py
        batch_size, revisits, channels, height, width = x.shape
        log2_revisits = torch.log2(torch.tensor(revisits))

        if log2_revisits % 1 > 0:

            nextpower = torch.ceil(log2_revisits)
            number_of_black_padding_revisits = int(2 ** nextpower - revisits)

            black_revisits = torch.zeros(
                batch_size,
                number_of_black_padding_revisits,
                channels,
                height,
                width,
                dtype=x.dtype,
                device=x.device,
            )

            x = torch.cat([x, black_revisits], dim=1)
        return x

    def forward(self, x):
        """ Forward pass of the RecursiveFusion layer.

        Parameters
        ----------
        x : Tensor
            The input tensor of shape (batch_size, revisits, in_channels, height, width).

        Returns
        -------
        Tensor
            The fused output tensor of shape (batch_size, in_channels, height, width).
        """
        x = self.pad(x)  # Zero-pad if neccessary to ensure power of 2 revisits
        x = self.fusion(x)  # (batch_size, 1, channels, height, width)
        return x.squeeze(1)  # (batch_size, channels, height, width)