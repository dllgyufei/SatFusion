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

class OneHot(nn.Module):
    """ One-hot encoder. """

    def __init__(self, num_classes):
        """ Initialize the OneHot layer.

        Parameters
        ----------
        num_classes : int
            The number of classes to encode.
        """
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        """ Forward pass.

        Parameters
        ----------
        x : Tensor
            The input tensor.

        Returns
        -------
        Tensor
            The encoded tensor.
        """
        # Current shape of x: (..., 1, H, W).
        x = x.to(torch.int64)
        # Remove the empty dimension: (..., H, W).
        x = x.squeeze(-3)
        # One-hot encode: (..., H, W, num_classes)
        x = F.one_hot(x, num_classes=self.num_classes)

        # Permute the dimensions so the number of classes is before the height and width.
        if x.ndim == 5:
            x = x.permute(0, 1, 4, 2, 3)  # (..., num_classes, H, W)
        elif x.ndim == 4:
            x = x.permute(0, 3, 1, 2)
        x = x.float()
        return x


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


class ResidualBlock(nn.Module):
    """ Two-layer 2D convolutional block (DoubleConv2d)
    with a skip-connection to a sum."""

    def __init__(self, in_channels, kernel_size=3, **kws):
        """ Initialize the ResidualBlock layer.

        Parameters
        ----------
        in_channels : int
            The number of input channels.
        kernel_size : int, optional
            The kernel size, by default 3.
        """
        super().__init__()
        self.residualblock = DoubleConv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            **kws,
        )

    def forward(self, x: Tensor) -> Tensor:
        """ Forward pass of the ResidualBlock layer.

        Parameters
        ----------
        x : Tensor
            The input tensor of shape (batch_size, in_channels, height, width).

        Returns
        -------
        Tensor
            The output tensor of shape (batch_size, in_channels, height, width).
        """
        x = x + self.residualblock(x)
        return x


class DenseBlock(ResidualBlock):
    """ Two-layer 2D convolutional block (DoubleConv2d) with a skip-connection
    to a concatenation (instead of a sum used in ResidualBlock)."""

    def forward(self, x: Tensor) -> Tensor:
        """ Forward pass of the DenseBlock layer.

        Parameters
        ----------
        x : Tensor
            The input tensor of shape (batch_size, in_channels, height, width).

        Returns
        -------
        Tensor
            The output tensor of shape (batch_size, in_channels, height, width).
        """
        return torch.cat([x, self.residualblock(x)], dim=1)


class FusionBlock(nn.Module):
    """ A block that fuses two revisits into one. """

    def __init__(self, in_channels, kernel_size=3, use_batchnorm=False):
        """ Initialize the FusionBlock layer.

        Fuse workflow:
        xx ---> xx ---> x
        |       ^
        |-------^

        Parameters
        ----------
        in_channels : int
            The number of input channels.
        kernel_size : int, optional
            The kernel size, by default 3.
        use_batchnorm : bool, optional
            Whether to use batch normalization, by default False.
        """
        super().__init__()

        # TODO: it might be better to fuse the encodings in groups - one per channel.

        number_of_revisits_to_fuse = 2

        self.fuse = nn.Sequential(
            # A two-layer 2D convolutional block with a skip-connection to a sum.
            ResidualBlock(
                number_of_revisits_to_fuse * in_channels,
                kernel_size,
                use_batchnorm=use_batchnorm,
            ),
            # A 2D convolutional layer.
            nn.Conv2d(
                in_channels=number_of_revisits_to_fuse * in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                padding="same",
                bias=not use_batchnorm,
                padding_mode="reflect",
            ),
            # Batch normalization, if requested.
            nn.BatchNorm2d(in_channels) if use_batchnorm else nn.Identity(),
            # Parametric ReLU activation.
            nn.PReLU(),
        )

    @staticmethod
    def split(x):
        """ Split the input tensor (revisits) into two parts/halves.

        Parameters
        ----------
        x : Tensor
            The input tensor of shape (batch_size, revisits, in_channels, height, width).

        Returns
        -------
        tuple of Tensor
            The two output tensors of shape (batch_size, revisits//2, in_channels, height, width).
        """

        number_of_revisits = x.shape[1]
        assert number_of_revisits % 2 == 0, f"number_of_revisits={number_of_revisits}"

        # (batch_size, revisits//2, in_channels, height, width)
        first_half = x[:, : number_of_revisits // 2].contiguous()
        second_half = x[:, number_of_revisits // 2 :].contiguous()

        # TODO: return a carry-encoding?
        return first_half, second_half

    def forward(self, x):
        """ Forward pass of the FusionBlock layer.


        Parameters
        ----------
        x : Tensor
            The input tensor of shape (batch_size, revisits, in_channels, height, width).
            Revisits encoding of the input.

        Returns
        -------
        Tensor
            The output tensor of shape (batch_size, revisits/2, in_channels, height, width).
            Fused encoding of the input.
        """

        first_half, second_half = self.split(x)
        batch_size, half_revisits, channels, height, width = first_half.shape

        first_half = first_half.view(
            batch_size * half_revisits, channels, height, width
        )

        second_half = second_half.view(
            batch_size * half_revisits, channels, height, width
        )

        # Current shape of x: (batch_size * revisits//2, 2*in_channels, height, width)
        x = torch.cat([first_half, second_half], dim=-3)

        # Fused shape of x: (batch_size * revisits//2, in_channels, height, width)
        fused_x = self.fuse(x)

        # Fused encoding shape of x: (batch_size, revisits/2, channels, width, height)
        fused_x = fused_x.view(batch_size, half_revisits, channels, height, width)

        return fused_x





class ConvTransposeBlock(nn.Module):
    """ Upsampler block with ConvTranspose2d. """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        sr_kernel_size,
        zoom_factor,
        use_batchnorm=False,
    ):
        """ Initialize the ConvTransposeBlock layer.

        Parameters
        ----------
        in_channels : int
            The number of input channels.
        out_channels : int
            The number of output channels.
        kernel_size : int
            The kernel size.
        sr_kernel_size : int
            The kernel size of the SR convolution.
        zoom_factor : int
            The zoom factor.
        use_batchnorm : bool, optional
            Whether to use batchnorm, by default False.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        # TODO: check if sr_kernel_size is the correct name
        self.sr_kernel_size = sr_kernel_size
        self.zoom_factor = zoom_factor
        self.use_batchnorm = use_batchnorm

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.in_channels,
                out_channels=self.in_channels,
                kernel_size=self.kernel_size,
                stride=self.zoom_factor,
                padding=0,
            ),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.in_channels,
                kernel_size=self.kernel_size,
                stride=1,
                padding="same",
                bias=not use_batchnorm,
                padding_mode="reflect",
            ),
            nn.BatchNorm2d(self.in_channels) if use_batchnorm else nn.Identity(),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.sr_kernel_size,
                stride=1,
                padding="same",
                bias=not use_batchnorm,
                padding_mode="reflect",
            ),
            nn.BatchNorm2d(self.out_channels) if use_batchnorm else nn.Identity(),
            nn.PReLU(),
        )

    def forward(self, x):
        """ Forward pass of the ConvTransposeBlock layer.

        Parameters
        ----------
        x : Tensor
            The input tensor of shape (batch_size, in_channels, height, width).

        Returns
        -------
        Tensor
            The output tensor of shape (batch_size, out_channels, height, width).
        """
        return self.upsample(x)


class PixelShuffleBlock(ConvTransposeBlock):

    """PixelShuffle block with ConvTranspose2d for sub-pixel convolutions. """

    # TODO: add a Dropout layer between the convolution layers?

    def __init__(self, **kws):
        super().__init__(**kws)
        assert self.in_channels % self.zoom_factor ** 2 == 0
        self.in_channels = self.in_channels // self.zoom_factor ** 2
        self.upsample = nn.Sequential(
            nn.PixelShuffle(self.zoom_factor),
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.in_channels,
                kernel_size=self.sr_kernel_size,
                stride=1,
                padding="same",
                bias=not self.use_batchnorm,
                padding_mode="reflect",
            ),
            nn.BatchNorm2d(self.in_channels) if self.use_batchnorm else nn.Identity(),
            nn.PReLU(),
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.sr_kernel_size,
                stride=1,
                padding="same",
                bias=not self.use_batchnorm,
                padding_mode="reflect",
            ),
            nn.BatchNorm2d(self.out_channels) if self.use_batchnorm else nn.Identity(),
            nn.PReLU(),
        )


class HomographyNet(nn.Module):
    """ A network that estimates a parametric geometric transformation matrix
    between two given images.

    Source: https://github.com/mazenmel/Deep-homography-estimation-Pytorch
    Reference: https://arxiv.org/pdf/1606.03798.pdf
    """

    def __init__(self, input_size, in_channels, fc_size, type="translation"):
        """ Initialize the HomographyNet layer.

        Parameters
        ----------
        input_size : tuple of int
            The input size.
        in_channels : int
            The number of input channels.
        fc_size : int
            The number of hidden channels.
        type : str, optional
            The type of transformation, by default 'translation'.

        Raises
        ------
        ValueError
            If the type of transformation is not supported.
        """

        super().__init__()

        if type not in ["translation", "homography"]:
            raise ValueError(
                f"Expected ['translation'|'homography'] for 'type'. Got {type}."
            )
        self.type = type
        self.input_size = input_size
        hidden_channels = fc_size // 2
        kernel_size = 3
        stride = 1
        padding = kernel_size // 2

        if self.kind == "translation":
            n_transform_params = 2
            self.transform = Shift()
        elif self.kind == "homography":
            n_transform_params = 8
            self.transform = WarpPerspective()

        def convblock(in_channels, out_channels, kernel_size, stride, padding):
            return nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    bias=False,
                    padding_mode="reflect",
                ),
                nn.BatchNorm2d(out_channels),
                nn.PReLU(),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    padding,
                    bias=False,
                    padding_mode="reflect",
                ),
                nn.BatchNorm2d(out_channels),
                nn.PReLU(),
            )

        self.cnn = nn.Sequential(
            convblock(
                in_channels=2 * in_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.MaxPool2d(2),
            convblock(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.MaxPool2d(2),
            convblock(
                in_channels=hidden_channels,
                out_channels=2 * hidden_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.MaxPool2d(2),
            convblock(
                in_channels=2 * hidden_channels,
                out_channels=2 * hidden_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
        )

        self.dropout = nn.Dropout(p=0.5)

        self.fc = nn.Sequential(
            nn.Linear(in_features=fc_size, out_features=n_transform_params, bias=False)
        )

    def register(self, image1, image2):
        """ Register two images.

        Parameters
        ----------
        image1 : Tensor
            The first image tensor of shape (batch_size, in_channels, height, width).
        image2 : Tensor
            The second image tensor of shape (batch_size, in_channels, height, width).

        Returns
        -------
        Tensor
            Parametric transformations on image1 with respect to image2 (batch_size, 2).
        """

        # Center each image by its own mean.
        image1 = image1 - torch.mean(image1, dim=(2, 3), keepdim=True)
        image2 = image2 - torch.mean(image2, dim=(2, 3), keepdim=True)

        # Concat channels (batch_size, 2 * channels, height, width)
        x = torch.cat([image1, image2], dim=1)

        x = self.cnn(x)

        # (batch_size, channels) global average pooling
        x = x.mean(dim=(2, 3))

        x = self.dropout(x)
        transformation_parameters = self.fc(x)
        transformation_parameters = torch.sigmoid(transformation_parameters) * 2 - 1

        return transformation_parameters

    def forward(self, image1, image2):
        """ Forward pass (register and transform two images).

        Parameters
        ----------
        image1 : Tensor
            The first image tensor of shape (batch_size, in_channels, height, width).
        image2 : Tensor
            The second image tensor of shape (batch_size, in_channels, height, width).

        Returns
        -------
        Tensor
            The transformed image tensor of shape (batch_size, in_channels, height, width).
        """
        with torch.autograd.set_detect_anomaly(True):
            transform_params = self.register(image1, image2)
            return self.transform(image1, transform_params)


class RFAB(nn.Module):
    """
    Residual Feature Attention Block.
    """
    def __init__(self, filters, kernel_size, r):
        super(RFAB, self).__init__()
        self.conv1 = nn.Conv3d(filters, filters, kernel_size, padding=1)
        self.conv2 = nn.Conv3d(filters, filters, kernel_size, padding=1)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(filters, filters // r, 1),
            nn.ReLU(),
            nn.Conv3d(filters // r, filters, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):                                           # 输入：(B, F, T, H, W)
        x_res = x
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        attn = self.attention(x)
        x = x * attn
        return x + x_res                                            # 输出：(B, F, T, H, W)

class RTAB(nn.Module):
    """
    Residual Temporal Attention Block.
    """
    def __init__(self, channel, kernel_size, r):
        super(RTAB, self).__init__()
        self.conv1 = nn.Conv2d(channel, channel, kernel_size, padding=1)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size, padding=1)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // r, 1),
            nn.ReLU(),
            nn.Conv2d(channel // r, channel, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):                                           # 输入：(B, T*C, H, W)
        x_res = x
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        attn = self.attention(x)
        x = x * attn
        return x + x_res

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5  # 1/sqrt(64)=0.125
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None, maps=None, K=64):
        b, n, _, h = *x.shape, self.heads  # b:batch_size, n:17, _:64, heads:heads as an example
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # self.to_qkv(x) to generate [b=batch_size, n=17, hd=192]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h),
                      qkv)  # q, k, v [b=batch_size, heads=heads, n=17, d=depth]
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale  # [b=batch_size, heads=heads, 17, 17]

        mask_value = -torch.finfo(dots.dtype).max  # A big negative number

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)  # [b=batch_size, 17]
            assert mask.shape[-1] == dots.shape[
                -1], 'mask has incorrect dimensions'  # mask [4, 17], dots [4, 8, 17, 17]
            assert len(mask.shape) == 2
            dots = dots.view(-1, K * K, dots.shape[1], dots.shape[2], dots.shape[3])
            mask = mask.unsqueeze(1).unsqueeze(2).unsqueeze(3)
            dots = dots * mask + mask_value * (1 - mask)
            dots = dots.view(-1, dots.shape[2], dots.shape[3], dots.shape[4])
            del mask

        if maps is not None:
            # maps [16384, 16] -> [16384, 17] , dots [16384, 8, 17, 17]
            maps = F.pad(maps.flatten(1), (1, 0), value=1.)
            maps = maps.unsqueeze(1).unsqueeze(2)
            dots.masked_fill_(~maps.bool(), mask_value)
            del maps

        attn = dots.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None, maps=None, K=64):
        for attn, ff in self.layers:
            x = attn(x, mask=mask, maps=maps, K=K)
            x = ff(x)
        return x

class SuperResTransformer_TRMISR(nn.Module):
    def __init__(self, *, dim, depth, heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        self.to_cls_token = nn.Identity()

    def forward(self, img, mask=None, maps=None, K=64):
        b, n, _ = img.shape
        # No need to add position code, just add token
        features_token = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((features_token, img), dim=1)
        x = self.transformer(x, mask, maps, K)
        x = self.to_cls_token(x[:, 0])

        return x

class ResidualBlock_TRMISR(nn.Module):
    def __init__(self, channel_size=64, kernel_size=3):
        '''
        Args:
            channel_size : int, number of hidden channels
            kernel_size : int, shape of a 2D kernel
        '''
        super(ResidualBlock_TRMISR, self).__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=channel_size, out_channels=channel_size, kernel_size=kernel_size, padding=padding),
            nn.PReLU(),
            nn.Conv2d(in_channels=channel_size, out_channels=channel_size, kernel_size=kernel_size, padding=padding),
            nn.PReLU()
        )

    def forward(self, x):
        '''
        Args:
            x : tensor (B, C, W, H), hidden state
        Returns:
            x + residual: tensor (B, C, W, H), new hidden state
        '''
        residual = self.block(x)
        return x + residual

class Encoder_TRMISR(nn.Module):
    def __init__(self, config):
        '''
        Args:
            config : dict, configuration file
        '''
        super(Encoder_TRMISR, self).__init__()

        in_channels = config["in_channels"]
        num_layers = config["num_layers"]
        kernel_size = config["kernel_size"]
        channel_size = config["channel_size"]
        padding = kernel_size // 2

        self.init_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=channel_size, kernel_size=kernel_size, padding=padding),
            nn.PReLU())

        res_layers = [ResidualBlock_TRMISR(channel_size, kernel_size) for _ in range(num_layers)]
        self.res_layers = nn.Sequential(*res_layers)

        self.final = nn.Sequential(
            nn.Conv2d(in_channels=channel_size, out_channels=channel_size, kernel_size=kernel_size, padding=padding)
        )

    def forward(self, x):
        '''
        Encodes an input tensor x.
        Args:
            x : tensor (B, C_in, W, H), input images
        Returns:
            out: tensor (B, C, W, H), hidden states
        '''
        x = self.init_layer(x)
        x = self.res_layers(x)
        x = self.final(x)
        return x

class Decoder_TRMISR(nn.Module):
    def __init__(self, config, out_channels):
        '''
        Args:
            config : dict, configuration file
        '''
        super(Decoder_TRMISR, self).__init__()

        self.final = nn.Sequential(nn.Conv2d(in_channels=config["final"]["in_channels"],
                                             out_channels=out_channels,
                                             kernel_size=config["final"]["kernel_size"],
                                             padding=config["final"]["kernel_size"] // 2),
                                   nn.PReLU())


    def forward(self, x):
        x = self.final(x)
        return x

###############################################################################
# Models
###############################################################################