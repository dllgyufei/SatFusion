import random
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import GaussianBlur
from torchvision.transforms import ColorJitter

# not use
class MultiTemporalGenerator:
    def __init__(self,
                 n_frames=8,
                 max_translation=0,
                 temporal_noise=0.0):
        """
        改进版多时相生成器 (带帧间差异)

        参数:
            temporal_noise: 帧间噪声强度 (默认0.00)
            temporal_jitter: 帧间亮度变化幅度 (默认±0%)
        """
        self.n_frames = n_frames
        self.max_translation = max_translation
        self.temporal_noise = temporal_noise

    def safe_random_translate(self, image, max_translation=0):
        """
        仅使用 pad + crop 实现平移，不插值，不引入亮度偏移。
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

        # 垂直方向
        if ty >= 0:
            y1_src, y2_src = 0, H - ty
            y1_dst, y2_dst = ty, H
        else:
            y1_src, y2_src = -ty, H
            y1_dst, y2_dst = 0, H + ty

        # 源区域 → 目标区域，尺寸严格匹配
        shifted[:, y1_dst:y2_dst, x1_dst:x2_dst] = image[:, y1_src:y2_src, x1_src:x2_src]

        return shifted

    def add_random_noise(self, image, noise_factor=0.00):
        """
        向图像添加随机噪声
        """
        std = image.std()
        adjusted_factor = std * noise_factor
        noise = torch.randn_like(image) * adjusted_factor
        noisy_image = image + noise
        # 确保像素值在[0, 1]之间
        noisy_image = torch.clamp(noisy_image, 0.0, 1.0)
        return noisy_image

    def generate(self, y: Tensor) -> Tensor:
        """
        生成带时序变化的低分辨率序列
        输入: [B, 1, C=8, h, w]
        输出: [B, R=8, C=8, h, w]
        """

        lr = y.repeat(1, self.n_frames, 1, 1, 1)
        # 对每张图片应用随机平移
        for i in range(lr.shape[1]):  # x.shape[1] 是 8 (批次中的图片数)
            for j in range(lr.shape[0]):  # x.shape[0] 是 2 (batch size)
                lr[j, i] = self.safe_random_translate(lr[j, i], self.max_translation)
                lr[j, i] = self.add_random_noise(lr[j, i], noise_factor=self.temporal_noise)
        return lr