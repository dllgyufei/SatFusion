import random
import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as TF
from torchvision.transforms import GaussianBlur
from torchvision.transforms.functional import InterpolationMode


class MultiTemporalGenerator_new:
    def __init__(self,
                 n_frames: int = 8,
                 target_size: tuple = (50, 50),
                 scale: int = 4,
                 psf_sigma_px: tuple = (1.5, 2.0),
                 shift_range: tuple = (-3, 3),
                 read_noise_std: float = 0.001,
                 shot_noise_gain: float = 1000,
                 temporal_noise: float = 0.02,
                 brightness_jitter: tuple = (0.9, 1.1)):
        """
        物理模拟多时相 LR-MS 数据生成器（WorldStrat）
        """
        self.n_frames = n_frames
        self.target_size = target_size
        self.scale = scale
        self.psf_sigma_px = psf_sigma_px
        self.shift_range = shift_range
        self.read_noise_std = read_noise_std
        self.shot_noise_gain = shot_noise_gain
        self.temporal_noise = temporal_noise
        self.brightness_jitter = brightness_jitter

    @staticmethod
    def _gauss_kernel1d(sigma, radius):
        x = torch.arange(-radius, radius+1, dtype=torch.float32)
        k = torch.exp(-0.5*(x/sigma)**2)
        k /= k.sum()
        return k

    def _separable_gauss(self, img, sigma):
        radius = max(1, int(3.0*sigma + 0.5))
        k1 = self._gauss_kernel1d(sigma, radius).to(img.device)
        kx = k1.view(1,1,1,-1)
        ky = k1.view(1,1,-1,1)
        B,C,H,W = img.shape
        pad = (radius, radius, radius, radius)
        img = F.pad(img, pad, mode='reflect')
        img = F.conv2d(img, kx.expand(C,1,1,kx.shape[-1]), groups=C)
        img = F.conv2d(img, ky.expand(C,1,ky.shape[-2],1), groups=C)
        return img

    def _warp_subpix(self, img, tx, ty):
        B,C,H,W = img.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1,1,H,device=img.device),
            torch.linspace(-1,1,W,device=img.device),
            indexing='ij'
        )
        gx = xx - 2.0*tx/(W-1)
        gy = yy - 2.0*ty/(H-1)
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(B,-1,-1,-1)
        return F.grid_sample(img, grid, mode='bilinear', padding_mode='border', align_corners=True)

    def _downsample(self, img):
        s = self.scale
        return F.avg_pool2d(img, kernel_size=s, stride=s)

    def _add_temporal_noise_and_brightness(self, lr):
        # 噪声
        if self.temporal_noise > 0:
            std = lr.std()
            lr = lr + torch.randn_like(lr) * std * self.temporal_noise
        # 亮度抖动
        factor = random.uniform(*self.brightness_jitter)
        lr = lr * factor
        return lr.clamp(0,1)

    def generate(self, hrms: torch.Tensor) -> torch.Tensor:
        """
        hrms: [B, C, H_hr, W_hr] 或 [B, 1, C, H_hr, W_hr]
        return: [B, R, C, h_lr, w_lr]
        """
        # 如果输入是 [B,1,C,H,W]，去掉中间的 1
        if hrms.dim() == 5 and hrms.shape[1] == 1:
            hrms = hrms[:,0]

        B,C,H,W = hrms.shape
        device = hrms.device
        dtype = hrms.dtype

        lr_stack = torch.zeros((B, self.n_frames, C, *self.target_size), device=device, dtype=dtype)

        for t in range(self.n_frames):
            # 随机亚像素位移
            tx = torch.empty(B, device=device).uniform_(*self.shift_range)
            ty = torch.empty(B, device=device).uniform_(*self.shift_range)
            warped = torch.stack([self._warp_subpix(hrms[b:b+1], tx[b], ty[b]) for b in range(B)], dim=0).squeeze(1)

            # 随机 MTF 模糊
            sigma = float(torch.empty(1).uniform_(*self.psf_sigma_px))
            blurred = self._separable_gauss(warped, sigma)

            # 下采样到 target_size
            lr = self._downsample(blurred)
            lr = F.interpolate(lr, size=self.target_size, mode='bilinear', align_corners=False)

            # 泊松-高斯噪声
            if self.shot_noise_gain > 0:
                photons = (lr * self.shot_noise_gain).clamp(min=0)
                noise = torch.randn_like(lr) * torch.sqrt(photons) / self.shot_noise_gain
                lr = lr + noise
            lr = lr + torch.randn_like(lr) * self.read_noise_std
            lr = lr.clamp(0,1)

            # temporal noise + brightness jitter
            #lr = self._add_temporal_noise_and_brightness(lr)

            lr_stack[:,t] = lr

        return lr_stack
