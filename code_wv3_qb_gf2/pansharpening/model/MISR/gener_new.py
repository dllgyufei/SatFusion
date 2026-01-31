import torch
import torch.nn.functional as F

class PhysMTMSynth:
    def __init__(self, n_frames=8, scale=4, psf_sigma_px=(1.2,1.8),
                 shift_range=(-1,1), read_noise_std=0.001, shot_noise_gain=5000):
        self.n_frames = n_frames
        self.scale = scale
        self.psf_sigma_px = psf_sigma_px  # 随机取值区间（像素）
        self.shift_range = shift_range    # 亚像素平移范围（HR 像素）
        self.read_noise_std = read_noise_std
        self.shot_noise_gain = shot_noise_gain

    @staticmethod
    def _gauss_kernel1d(sigma, radius):
        x = torch.arange(-radius, radius+1, dtype=torch.float32)
        k = torch.exp(-0.5*(x/sigma)**2)
        k /= k.sum()
        return k

    def _separable_gauss(self, img, sigma):
        # img: [B,C,H,W] (HR)
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
        # 连续位移，grid_sample 使用 bilinear
        B,C,H,W = img.shape
        yy, xx = torch.meshgrid(
            torch.linspace(-1,1,H,device=img.device),
            torch.linspace(-1,1,W,device=img.device),
            indexing='ij'
        )
        # 以 HR 像素为单位的位移 -> 归一化坐标
        gx = xx - 2.0*tx/(W-1)
        gy = yy - 2.0*ty/(H-1)
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(B,-1,-1,-1)
        return F.grid_sample(img, grid, mode='bilinear', padding_mode='border', align_corners=True)

    def _downsample(self, img):
        # 先模糊后下采样；可改为 Area Pooling
        s = self.scale
        return F.avg_pool2d(img, kernel_size=s, stride=s)

    def generate(self, hrms: torch.Tensor) -> torch.Tensor:
        """
        hrms: [B,C,H,W]  (原始 HR-MS)
        return: [B, R, C, H_lr, W_lr]  (合成的多时相 LR-MS stack)
        """
        B,C,H,W = hrms.shape
        frames = []
        for _ in range(self.n_frames):
            # 1) 随机亚像素位移
            tx = torch.empty(B, device=hrms.device).uniform_(*self.shift_range)
            ty = torch.empty(B, device=hrms.device).uniform_(*self.shift_range)
            warped = torch.stack(
                [self._warp_subpix(hrms[b:b+1], tx[b], ty[b]) for b in range(B)],
                dim=0
            ).squeeze(1)

            # 2) 传感器 MTF 模糊
            sigma = float(torch.empty(1).uniform_(*self.psf_sigma_px))
            blurred = self._separable_gauss(warped, sigma)

            # 3) 下采样
            lr = self._downsample(blurred).clamp(0,1)

            # 4) 泊松-高斯噪声
            if self.shot_noise_gain > 0:
                photons = (lr * self.shot_noise_gain).clamp(min=0)
                # 用 Gaussian 近似代替 poisson 抽样，更平滑
                noise = torch.randn_like(lr) * torch.sqrt(photons) / self.shot_noise_gain
                shot = lr + noise
            else:
                shot = lr

            lr_noisy = shot + torch.randn_like(lr) * self.read_noise_std
            frames.append(lr_noisy.clamp(0,1))

        lr_stack = torch.stack(frames, dim=1)  # [B,R,C,h,w]
        return lr_stack
