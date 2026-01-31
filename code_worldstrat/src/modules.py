from typing_extensions import Self
from src.sharp.psit.GPPNN_PSIT import GPPNN_With_MISR as GPPNN_PSIT_With_MISR
from src.sharp.psit.GPPNN_PSIT import GPPNN_Only_Sharpening as GPPNN_PSIT_Only_Sharpening
from src.sharp.pnn.DoubleConv2d import DoubleConv2d_PNN
from src.sharp.pnn.pnn import PCNN
from src.sharp.pannet.pannet import PANNet_With_MISR, PANNet_Only_Sharpening
from src.sharp.u2net.u2net import U2Net_With_MISR, U2Net_Only_Sharpening
# from src.sharp.mamba.panmamba_baseline_finalversion import Net_With_MISR, Net_Only_Sharpening
from src.sharp.ARConv.model import ARNet_With_MISR, ARNet_Only_Sharpening
from src.misr.srcnn import SRCNN
from src.misr.highresnet import RecursiveFusion
from src.misr.trnet import TRNet
from src.misr.rams import RAMS
from src.misr.misr_public_modules import DoubleConv2d
from src.misr.misr_public_modules import PixelShuffleBlock
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
import random
from torchvision.transforms import GaussianBlur
from torchvision.transforms import ColorJitter
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode


###############################################################################
# Layers
###############################################################################

class ColorCorrection(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        # ​​1x1 Conv(without changing H/W)​
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1),  # expand channel
            nn.ReLU(inplace=True),
            nn.Conv2d(64, in_channels, kernel_size=1)  # recover channel
        )

    def forward(self, x):
        """
        input: x [B,C,H,W]
        outout: [B,C,H,W]
        """
        return x + self.conv(x)  # ​​Residual Connection: Original Image + Learned Color Offset​


def replace_furthest_with_median(images):
    """
    Args:
        images: (B, R, C, H, W) tensor

    Returns:
        (B, R, C, H, W) tensor with furthest image replaced by median
    """
    B, R, C, H, W = images.shape

    # Compute median image (B, C, H, W)​
    median_img = torch.median(images, dim=1).values

    # ​​Compute the mean value of each channel for every image (B, R, C)​
    img_means = images.mean(dim=[-2, -1])  # (B, R, C)
    median_means = median_img.mean(dim=[-2, -1])  # (B, C)

    # ​​Compute Euclidean distance (B, R)​
    distances = torch.norm(img_means - median_means.unsqueeze(1), dim=2)

    # Argmax of distances per batch (B,)​
    furthest_idx = torch.argmax(distances, dim=1)

    # ​​Initialize output tensor with in-place replacement​
    output = images.clone()
    batch_indices = torch.arange(B)
    output[batch_indices, furthest_idx] = median_img.detach()

    return output


class MultiTemporalGenerator:
    def __init__(self,
                 n_frames=8,
                 target_size: tuple = (50, 50),
                 kernel_size=5,
                 sigma=0.0,
                 temporal_noise=0.02,
                 temporal_jitter=0.1):
        """
        params:
            temporal_noise:Inter-frame noise intensity (default: 0.02)
            temporal_jitter:Inter-frame brightness variation range (default: ±10%)
        """
        self.n_frames = n_frames
        self.target_size = target_size
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.temporal_noise = temporal_noise
        self.temporal_jitter = temporal_jitter

    def safe_random_translate(self, image, max_translation=4):
        """
        Achieve Image Translation via Padding and Cropping Operations
        image: [C, H, W]
        """
        tx = random.randint(-max_translation, max_translation)
        ty = random.randint(-max_translation, max_translation)
        C, H, W = image.shape
        shifted = torch.zeros_like(image)

        if tx >= 0:
            x1_src, x2_src = 0, W - tx
            x1_dst, x2_dst = tx, W
        else:
            x1_src, x2_src = -tx, W
            x1_dst, x2_dst = 0, W + tx

        if ty >= 0:
            y1_src, y2_src = 0, H - ty
            y1_dst, y2_dst = ty, H
        else:
            y1_src, y2_src = -ty, H
            y1_dst, y2_dst = 0, H + ty

        shifted[:, y1_dst:y2_dst, x1_dst:x2_dst] = image[:, y1_src:y2_src, x1_src:x2_src]

        return shifted

    def random_translate(self, image, max_translation=4):
        translate_x = random.randint(-max_translation, max_translation)
        translate_y = random.randint(-max_translation, max_translation)

        # Image Translation via Affine Transformation in TorchVision​
        image = TF.affine(
            image,
            angle=0,
            translate=(translate_x, translate_y),
            scale=1.0,
            shear=0,
            interpolation=InterpolationMode.BILINEAR,
            fill=[0.0, 0.0, 0.0]
        )
        return image

    def add_random_noise(self, image, noise_factor=0.03):
        """
        add Random Noise to Images
        """
        std = image.std()
        adjusted_factor = std * noise_factor
        noise = torch.randn_like(image) * adjusted_factor
        noisy_image = image + noise
        noisy_image = torch.clamp(noisy_image, 0.0, 1.0)
        return noisy_image

    def adjust_random_brightness(self, image, brightness_factor_range=(0.9, 1.1)):
        """
        Apply Randomized Brightness Perturbation to Images
        """
        jitter = ColorJitter(brightness=brightness_factor_range)
        return jitter(image)

    def generate(self, y: Tensor) -> Tensor:
        """
        Synthesize Multi-Frame Low-Res Sequences with Temporal Dynamics
        input: [B, 1, C=3, H, W]
        output: [B, R=8, C=3, h, w] (h=H//scale_factor)
        """
        gaussian_blur = GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))

        x_blurred = gaussian_blur(y[:, 0])

        x_downsampled = F.interpolate(x_blurred, size=self.target_size, mode='bilinear', align_corners=False)
        x_downsampled = x_downsampled.unsqueeze(1)

        lr = x_downsampled.repeat(1, self.n_frames, 1, 1, 1)

        for i in range(lr.shape[1]):
            for j in range(lr.shape[0]):
                lr[j, i] = self.safe_random_translate(lr[j, i],2)
                lr[j, i] = self.add_random_noise(lr[j, i], noise_factor=self.temporal_noise)
                lr[j, i] = self.adjust_random_brightness(lr[j, i], brightness_factor_range=(0.9, 1.1))
                # lr[j, i] = self.safe_random_translate(lr[j, i], 2)
                # lr[j, i] = self.add_random_noise(lr[j, i], noise_factor=0.1 * self.temporal_jitter)
                # lr[j, i] = self.adjust_random_brightness(lr[j, i], brightness_factor_range=(
                # 1 - 0.05 * self.temporal_jitter, 1 + 0.05 * self.temporal_jitter))
        return lr


###############################################################################
# OurModels
###############################################################################
class OurMISR(nn.Module):
    """
    OurMISR model : 实现MISR过程中，除了采样之外的其他步骤
    """

    def __init__(
            self,
            model_MISR,
            msi_channels,
            revisits,
            hidden_channels,
            out_channels,
            kernel_size,
            output_size,
            zoom_factor,
            sr_kernel_size,
            use_artificial_dataset,
            use_sampling_model,
            **kws,
    ) -> None:

        super().__init__()
        self.model_MISR = model_MISR
        self.in_channels = msi_channels  # use reference graph
        self.revisits = revisits
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.output_size = output_size
        self.zoom_factor = zoom_factor
        self.sr_kernel_size = sr_kernel_size
        self.use_batchnorm = False
        self.use_artificial_dataset = use_artificial_dataset
        self.use_sampling_model = use_sampling_model

        if self.model_MISR in ['SRCNN']:
            self.encoder = DoubleConv2d(
                in_channels=2 * self.in_channels,
                out_channels=hidden_channels,
                kernel_size=self.kernel_size,
                use_batchnorm=self.use_batchnorm,
            )
        if self.model_MISR in ['HighResNet']:
            self.encoder = DoubleConv2d(
                in_channels=self.in_channels,
                out_channels=hidden_channels,
                kernel_size=self.kernel_size,
                use_batchnorm=self.use_batchnorm,
            )

        # Fusion SRCNN
        if self.model_MISR in ['SRCNN']:
            self.blockMISR = SRCNN(
                residual_layers=1,
                hidden_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=self.kernel_size,
                revisits=self.revisits,
            )

        # Fusion HighResNet
        if self.model_MISR in ['HighResNet']:
            self.blockMISR = RecursiveFusion(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=self.kernel_size,
                revisits=self.revisits,
            )

        # Fusion RAMS
        if self.model_MISR == 'RAMS':
            self.blockMISR = RAMS(
                scale=self.zoom_factor,
                filters=32,  # number of 3D Convolution
                kernel_size=self.kernel_size,
                depth=self.revisits,
                r=8,  # Compression Rate in Attention Mechanisms
                N=12,  # Number of residual feature attention blocks
                out_channels=self.hidden_channels,
            )

        # Fusion TRNet
        if self.model_MISR == 'TRNet':
            self.blockMISR = TRNet(
                config={
                    "encoder": {
                        "in_channels": self.in_channels,
                        "num_layers": 2,
                        "kernel_size": 3,
                        "channel_size": 64
                    },
                    "transformer": {
                        "dim": 64,  # Transformer Dimensions
                        "depth": 6,  # Transformer depth
                        "heads": 8,  # Head count in multi-head self-attention
                        "mlp_dim": 128,  # FFN Intermediate Dimension
                        "dropout": 0.1  # Dropout ​​Probability​
                    },
                    "decoder": {
                        "final": {
                            "in_channels": 64,
                            "kernel_size": 3
                        }
                    }
                },
                out_channels=self.hidden_channels,
                use_pan=False,
            )

        # Fusion SatFusion*
        if self.model_MISR == 'TRNet_pan':
            self.pan_encoder = nn.Sequential(
                nn.Conv2d(1, msi_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(msi_channels, msi_channels, kernel_size=3, padding=1),
            )
        if self.model_MISR == 'TRNet_pan':
            self.blockMISR = TRNet(
                config={
                    "encoder": {
                        "in_channels": self.in_channels + 1,  # 在channel维度上拼接pan
                        "num_layers": 2,
                        "kernel_size": 3,
                        "channel_size": 64
                    },
                    "transformer": {
                        "dim": 64,  # Transformer Dimensions
                        "depth": 6,  # Transformer depth
                        "heads": 8,  # Head count in multi-head self-attention
                        "mlp_dim": 128,  # FFN Intermediate Dimension
                        "dropout": 0.1  # Dropout ​​Probability​
                    },
                    "decoder": {
                        "final": {
                            "in_channels": 64,
                            "kernel_size": 3
                        }
                    }
                },
                out_channels=self.hidden_channels,
                use_pan=True,
            )

        ## Super-resolver (upsampler + renderer)
        self.sr = PixelShuffleBlock(
            in_channels=self.hidden_channels,
            out_channels=self.out_channels,
            kernel_size=self.kernel_size,
            sr_kernel_size=self.sr_kernel_size,
            zoom_factor=self.zoom_factor,
            use_batchnorm=self.use_batchnorm,
        )

        self.resize = Resize(
            self.output_size,
            interpolation="bilinear",
            align_corners=False,
            antialias=True,
        )

    def forward(
            self, x: Tensor, pan: Optional[Tensor] = None, y: Optional[Tensor] = None,
    ) -> Tensor:
        if self.use_sampling_model:  # filter
            x = replace_furthest_with_median(x)

        # pan 的 shape 是 [B, 1, 1, H, W]
        if self.model_MISR in ['TRNet_pan']:
            _, _, _, h_low, w_low = x.shape  # MISR 输入分辨率
            pan = pan.squeeze(1)
            pan = F.interpolate(pan, size=(h_low, w_low), mode="bilinear", align_corners=False)
            pan_expand = pan.unsqueeze(1).expand(-1, self.revisits, -1, -1, -1)
            x = torch.cat([x, pan_expand], dim=2)

        if self.model_MISR == 'SRCNN':
            x = self.compute_and_concat_reference_frame_to_input(x)

        # transform to [B*R,C,H,W] and encoder (except for RAMS and TRNet)
        batch_size, revisits, channels, height, width = x.shape
        hidden_channels = self.hidden_channels
        if self.model_MISR in ['SRCNN', 'HighResNet']:
            x = x.view(batch_size * revisits, channels, height, width)
            x = self.encoder(x)

            # Preprocessing
        if self.model_MISR == 'SRCNN':
            x = x.view(batch_size, revisits * hidden_channels, height, width)
        elif self.model_MISR == 'HighResNet':
            x = x.view(batch_size, revisits, hidden_channels, height, width)
        elif self.model_MISR == 'RAMS':
            x = x.view(batch_size, revisits, 3, height, width)
            x = x.permute(0, 2, 1, 3, 4)
        elif self.model_MISR == 'TRNet':
            x = x.view(batch_size, revisits, channels, height, width)
        elif self.model_MISR == 'TRNet_pan':
            x = x.view(batch_size, revisits, channels, height, width)

        # fusion
        if self.model_MISR == 'SRCNN':
            x = self.blockMISR(x)
        elif self.model_MISR == 'HighResNet':
            x = self.blockMISR(x)
        elif self.model_MISR == 'RAMS':
            x = self.blockMISR(x)
        elif self.model_MISR in ['TRNet_pan']:
            pan_feat = self.pan_encoder(pan)  # [B, C, H, W]
            x = self.blockMISR(x, pan_feat=pan_feat, K=50)
        elif self.model_MISR in ['TRNet']:
            x = self.blockMISR(x, pan_feat=None, K=50)
        '''
        补充其他融合方法
        '''

        # Upsampling
        x = self.sr(x)

        # resize
        _, _, H_x, W_x = x.size()
        if (H_x, W_x) != self.output_size:
            x = self.resize(x)

        x = x[:, None]

        return x  # [B,1,3,H,W]

    def compute_and_concat_reference_frame_to_input(self, x):
        # Current shape: (batch_size, revisits, channels, height, width)
        reference_frame = self.reference_frame(x).expand_as(x)
        # Concatenated shape: (batch_size, revisits, 2*channels, height, width)
        x = torch.cat([x, reference_frame], dim=-3)
        return x

    def reference_frame(self, x: Tensor) -> Tensor:
        """ Compute the reference frame as the median of all low-res revisits.

        Parameters
        ----------
        x : Tensor
            The input tensor (low-res revisits).

        Returns
        -------
        Tensor
            The reference frame.
        """
        return x.median(dim=-4, keepdim=True).values


class OurSharpening(nn.Module):
    """
        OurSharpening model : 实现Sharpening过程
    """

    def __init__(
            self,
            model_Sharpening,
            msi_channels,
            pan_channels,
            kernel_size,
            output_size,
            out_channels,
            only_use_oursharpening,
            use_artificial_dataset,
            **kws,
    ) -> None:
        ''' 参数解释
        model_Sharpening: the model of sharpening like "PNN"
        model_MISR: the model of MISR like "SRCNN"
        in_channels: the of output channel numbers of MISR
        out_channels: the final output channel numbers
        '''
        super().__init__()

        self.only_use_oursharpening = only_use_oursharpening
        self.model_Sharpening = model_Sharpening
        self.msi_channels = msi_channels
        self.pan_channels = pan_channels
        self.kernel_size = kernel_size
        self.use_batchnorm = False
        self.output_size = output_size
        self.out_channels = out_channels
        self.use_artificial_dataset = use_artificial_dataset

        self.resize = Resize(
            self.output_size,
            interpolation="bilinear",
            align_corners=False,
            antialias=True,
        )

        if self.model_Sharpening == 'PNN':
            self.blockSharp = PCNN(
                in_channels=self.msi_channels + self.pan_channels,
                # params refer to related paper(PNN)
                n1=64,  # the number of first convolution​​
                n2=32,  # the number of second convolution​​
                f1=9,  # the kernel size of first convolution​​
                f2=5,  # the kernel size of second convolution​​
                f3=5,  # the kernel size of output convolution​​
            )
        if self.only_use_oursharpening == False and self.model_Sharpening == 'PANNet':
            self.blockSharp = PANNet_With_MISR(
                in_channels=self.msi_channels + self.pan_channels,
                n1=64,
                n2=32,
                f1=9,
                f2=5,
                f3=5,
            )
        if self.only_use_oursharpening and self.model_Sharpening == 'PANNet':
            self.blockSharp = PANNet_Only_Sharpening(
                in_channels=self.msi_channels + self.pan_channels,
                n1=64,
                n2=32,
                f1=9,
                f2=5,
                f3=5,
                output_size=self.output_size,
            )
        if self.only_use_oursharpening==False and self.model_Sharpening=='U2Net':
            self.blockSharp = U2Net_With_MISR(
                dim = 32,
            )
        if self.only_use_oursharpening and self.model_Sharpening=='U2Net':
            self.blockSharp = U2Net_Only_Sharpening(
                dim = 32,
            )
        if self.only_use_oursharpening==False and self.model_Sharpening=='PSIT':
            self.blockSharp = GPPNN_PSIT_With_MISR(
                ms_channels=self.msi_channels,
                pan_channels=self.pan_channels,
                n_feat=16,
            )
        if self.only_use_oursharpening and self.model_Sharpening=='PSIT':
            self.blockSharp = GPPNN_PSIT_Only_Sharpening(
                ms_channels=self.msi_channels,
                pan_channels=self.pan_channels,
                n_feat=16,
            )
        if self.only_use_oursharpening==False and self.model_Sharpening=='Pan_Mamba':
            self.blockSharp = Net_With_MISR(
                base_filter=32
            )
        if self.only_use_oursharpening and self.model_Sharpening=='Pan_Mamba':
            self.blockSharp = Net_Only_Sharpening(
                base_filter=32
            )
        if self.only_use_oursharpening==False and self.model_Sharpening=='ARConv':
            self.blockSharp = ARNet_With_MISR(
                pan_channels=1,
                lms_channels=self.msi_channels,
            )
        if self.only_use_oursharpening and self.model_Sharpening=='ARConv':
            self.blockSharp = ARNet_Only_Sharpening(
                pan_channels=1,
                lms_channels=self.msi_channels,
            )

    def forward(
            self, x: Tensor, pan: Tensor, y: Optional[Tensor] = None,
    ) -> Tensor:
        if self.only_use_oursharpening == False:  # x is the output of MISR
            x = x.squeeze(1)  # [B, C, H, W]
            pan = pan.squeeze(1)  # [B, 1, H, W]

            if self.model_Sharpening == 'PNN':
                x = self.blockSharp(ms=x, pan=pan)
            if (self.model_Sharpening == 'PANNet'):
                x = self.blockSharp(ms=x, pan=pan)
            if (self.model_Sharpening == 'Pan_Mamba'):
                x = self.blockSharp.forward(ms=x, _=0, pan=pan)
            if (self.model_Sharpening == 'PSIT'):
                x = self.blockSharp(ms=x, pan=pan)
            if (self.model_Sharpening == 'U2Net'):
                x = self.blockSharp(x=x, y=pan)
            if (self.model_Sharpening == 'ARConv'):
                x = self.blockSharp(lms=x, pan=pan, epoch=None, hw_range=[1,9])

            x = x[:, None]
            return x
        else:  # choose a gragh randomly from ms
            pan = pan.squeeze(1)  # [B, 1, H, W]
            i = random.randint(0, x.shape[1] - 1)
            # print(i)
            x = x[:, i, :, :, :]
            # fusion
            if self.model_Sharpening == 'PNN':
                x = self.resize(x)
                x = self.blockSharp(ms=x, pan=pan)
            if (self.model_Sharpening == 'PANNet'):
                x = self.blockSharp(ms=x, pan=pan)
            if (self.model_Sharpening == 'Pan_Mamba'):
                x = self.resize(x)
                x = self.blockSharp.forward(ms=x, _=0, pan=pan)
            if (self.model_Sharpening == 'PSIT'):
                x = self.resize(x)
                x = self.blockSharp(ms=x, pan=pan)
            if (self.model_Sharpening == 'U2Net'):
                x = self.resize(x)
                x = self.blockSharp(x=x, y=pan)
            if (self.model_Sharpening == 'ARConv'):
                x = self.resize(x)
                x = self.blockSharp(lms=x, pan=pan, epoch=None, hw_range=[1,9])
            x = x[:, None]

            return x


class OurFramework(nn.Module):
    """
        OurFramework model : the complete process
    """

    def __init__(
            self,
            model_MISR,
            model_Sharpening,
            msi_channels,
            MISR_revisits,
            MISR_hidden_channels,
            MISR_kernel_size,
            output_size,
            out_channels,
            MISR_zoom_factor,
            MISR_sr_kernel_size,
            pan_channels,
            Sharpening_kernel_size,
            use_artificial_dataset,
            use_sampling_model,
            **kws,
    ) -> None:
        super().__init__()
        self.model_MISR = model_MISR
        self.model_Sharpening = model_Sharpening
        self.msi_channels = msi_channels
        self.MISR_revisits = MISR_revisits
        self.MISR_hidden_channels = MISR_hidden_channels
        self.MISR_kernel_size = MISR_kernel_size
        self.output_size = output_size
        self.MISR_zoom_factor = MISR_zoom_factor
        self.MISR_sr_kernel_size = MISR_sr_kernel_size
        self.pan_channels = pan_channels
        self.out_channels = out_channels
        self.Sharpening_kernel_size = Sharpening_kernel_size
        self.use_artificial_dataset = use_artificial_dataset
        self.use_sampling_model = use_sampling_model

        # create MISR class
        self.MISR = OurMISR(
            model_MISR=self.model_MISR,
            msi_channels=self.msi_channels,
            out_channels=out_channels,
            revisits=self.MISR_revisits,
            hidden_channels=self.MISR_hidden_channels,
            kernel_size=self.MISR_kernel_size,
            output_size=self.output_size,
            zoom_factor=self.MISR_zoom_factor,
            sr_kernel_size=self.MISR_sr_kernel_size,
            use_artificial_dataset=self.use_artificial_dataset,
            use_sampling_model=self.use_sampling_model,
        )

        # create Sharpening Class
        self.Sharpening = OurSharpening(
            model_Sharpening=self.model_Sharpening,
            msi_channels=self.msi_channels,
            pan_channels=self.pan_channels,
            kernel_size=self.Sharpening_kernel_size,
            output_size=self.output_size,
            out_channels=self.out_channels,
            use_artificial_dataset=self.use_artificial_dataset,
            only_use_oursharpening=True if self.model_MISR == "None" else False
        )

        self.color_correct = ColorCorrection()

    def forward(
            self, x: Tensor, pan: Tensor, y: Optional[Tensor] = None
    ):
        # initialize the intermediate output for computing the loss ​multi-stage​​dly
        misr_out, sharpening_out = None, None

        # Stage 1: MISR
        if self.model_MISR != "None":
            misr_out = self.MISR(x=x, pan=pan, y=y)  # current_out = misr_out: [B, 1, C=3, H, W]
            current_out = misr_out
        else:
            current_out = x  # current_out = x: [B, R, C=3, h, w]

        # Stage 2、3: Sharpening、Combination
        if self.model_Sharpening != "None" and self.model_MISR != "None":  # MISR+Sharpening
            sharpening_out = self.Sharpening(x=current_out, pan=pan, y=y)
            current_out = current_out + sharpening_out
            current_out = current_out.squeeze(1)
            current_out = self.color_correct(current_out)
            current_out = current_out[:, None]

        elif self.model_Sharpening != "None" and self.model_MISR == "None":  # None+Sharpening
            sharpening_out = self.Sharpening(x=current_out, pan=pan, y=y)
            current_out = sharpening_out
        else:  # MISR+None
            current_out = current_out.squeeze(1)
            current_out = current_out[:, None]
        return current_out, misr_out, sharpening_out


if __name__ == "__main__":
    model_MISR = "RAMS"
    model_Sharpening = "None"
    MISR_in_channels = 3
    MISR_revisits = 8
    MISR_hidden_channels = 128
    MISR_out_channels = 3
    MISR_kernel_size = 3
    output_size = (156, 156)
    MISR_zoom_factor = 2
    MISR_sr_kernel_size = 1
    MISR_registration_kind = 'homography'
    MISR_homography_fc_size = 128
    Sharpening_in_channels = 3
    Sharpening_hidden_channels = 128
    out_channels = 3
    Sharpening_kernel_size = 3
    MISR_use_reference_frame = True

    model = OurFramework(
        model_MISR=model_MISR,
        model_Sharpening=model_Sharpening,
        MISR_in_channels=MISR_in_channels,
        MISR_revisits=MISR_revisits,
        MISR_hidden_channels=MISR_hidden_channels,
        MISR_kernel_size=MISR_kernel_size,
        output_size=output_size,
        MISR_zoom_factor=MISR_zoom_factor,
        MISR_sr_kernel_size=MISR_sr_kernel_size,
        MISR_registration_kind=MISR_registration_kind,
        MISR_homography_fc_size=MISR_homography_fc_size,
        Sharpening_in_channels=Sharpening_in_channels,
        Sharpening_hidden_channels=Sharpening_hidden_channels,
        out_channels=out_channels,
        Sharpening_kernel_size=Sharpening_kernel_size,
        MISR_use_reference_frame=MISR_use_reference_frame,
    )

    batch_size = 2
    height, width = 50, 50  # shape of ms
    high_height, high_width = 156, 156  # shape of pan and hr
    x = torch.rand(batch_size, MISR_revisits, MISR_in_channels, height, width, dtype=torch.float32)
    pan = torch.rand(batch_size, 1, 1, high_height, high_width, dtype=torch.float32)
    print(f"x:", x.shape)
    print(f"pan:", pan.shape)

    # check the output
    SR = model(x=x, pan=pan)

    print(f"y:", SR.shape)


