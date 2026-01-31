from typing_extensions import Self
from src.datasources import JIF_S2_MEAN, JIF_S2_STD, S2_ALL_12BANDS
import numpy as np
import torch
import torch.nn.functional as F
from math import log2
from kornia.geometry.transform import Resize
from torch import nn, Tensor
from typing import Tuple, Optional
from src.transforms import Shift, WarpPerspective
import math
from einops import rearrange, repeat

class DoubleConv2d(nn.Module):
    """ Two-layer 2D convolutional block with a PReLU activation in between. """

    # TODO: verify if we still need reflect padding-mode. If we do, set via constructor.

    def __init__(self, in_channels, out_channels, kernel_size=3, use_batchnorm=False):
        """ Initialize the DoubleConv2d layer.

        Parameters
        ----------
        in_channels : int
            The number of input channels.
        out_channels : int
            The number of output channels.
        kernel_size : int, optional
            The kernel size, by default 3.
        use_batchnorm : bool, optional
            Whether to use batch normalization, by default False.
        """
        super().__init__()

        self.doubleconv2d = nn.Sequential(
            # ------- First block -------
            # First convolutional layer.
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding="same",
                bias=not use_batchnorm,
                padding_mode="reflect",
            ),
            # Batch normalization, if requested.
            nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
            # Parametric ReLU activation.
            nn.PReLU(),
            # Dropout regularization, keep probability 0.5.
            nn.Dropout(p=0.5),
            # ------- Second block -------
            # Second convolutional layer.
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding="same",
                bias=not use_batchnorm,
                padding_mode="reflect",
            ),
            # Batch normalization, if requested.
            nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity(),
            # Parametric ReLU activation.
            nn.PReLU(),
            # Dropout regularization, keep probability 0.5.
            nn.Dropout(p=0.5),
        )
        
    def forward(self, x: Tensor) -> Tensor:
        """ Forward pass of the DoubleConv2d layer.

        Parameters
        ----------
        x : Tensor
            The input tensor of shape (batch_size, in_channels, height, width).

        Returns
        -------
        Tensor
            The output tensor of shape (batch_size, out_channels, height, width).
        """
        return self.doubleconv2d(x)
    
class DoubleConv2d_PNN(nn.Module):
    """ Super-resolution CNN.
    Uses no recursive function, revisits are treated as channels.
    """

    def __init__(
        self,
        hidden_channels,
        in_channels,
        out_channels,
        kernel_size,
        **kws,
    ) -> None:
        super().__init__()

        self.doubleconv2d_first = DoubleConv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
        )  
        self.doubleconv2d_second = DoubleConv2d(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
        )
        self.model_Double_DoubleConv2d = nn.Sequential(self.doubleconv2d_first, self.doubleconv2d_second)

    def forward(
        self, x: Tensor, y: Optional[Tensor] = None, pan: Optional[Tensor] = None, mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = self.model_Double_DoubleConv2d(x)
        return x