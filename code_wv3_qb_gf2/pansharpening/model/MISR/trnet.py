from typing_extensions import Self
import numpy as np
import torch
import torch.nn.functional as F
from .misr_public_modules import Encoder_TRMISR
from .misr_public_modules import SuperResTransformer_TRMISR
from .misr_public_modules import Decoder_TRMISR
from math import log2
from kornia.geometry.transform import Resize
from torch import nn, Tensor
from typing import Tuple, Optional
import math
from einops import rearrange, repeat


class TRNet(nn.Module):
    ''' TRNet, a neural network for multi-frame super resolution (MFSR) by recursive fusion. '''

    def __init__(self, config, out_channels):

        super(TRNet, self).__init__()
        self.encode = Encoder_TRMISR(config["encoder"])
        self.superres = SuperResTransformer_TRMISR(dim=config["transformer"]["dim"],
                                            depth=config["transformer"]["depth"],
                                            heads=config["transformer"]["heads"],
                                            mlp_dim=config["transformer"]["mlp_dim"],
                                            dropout=config["transformer"]["dropout"])
        self.decode = Decoder_TRMISR(config["decoder"], out_channels)
        self.out_channels = out_channels

    def forward(self, x, K, pan_feat=None, alphas=None, maps=None):
        batch_size, revisits, channels, height, width = x.shape
        x = x.view(batch_size*revisits, channels, height, width)

        ####################### encode ######################
        x = self.encode(x)
        x = x.view(batch_size, revisits, -1, height, width)

        ####################### fuse ######################
        x = x.permute(0, 3, 4, 1, 2).reshape(-1, x.shape[1],
                                               x.shape[2])

        if pan_feat != None:
            x = self.superres(x, K=K, pan_feat = pan_feat)
        else :
            x = self.superres(x, K=K, pan_feat = None)
        x = x.view(-1, K, K, x.shape[-1]).permute(0, 3, 1, 2)

        ####################### decode ######################
        x = self.decode(x)  # 输出数据是 (B, hidden_channels, H, W)

        return x