import torch
from torch import nn, Tensor
class PCNN(nn.Module):
    """
    PCNN for Pansharpening Task
    Modified to match WorldStrat's input/output requirements
    Based on the original PCNN architecture from Masi et al. (2016)
    """

    def __init__(self, in_channels=4, n1=64, n2=32, f1=9, f2=5, f3=5):
        super().__init__()


        self.conv1 = nn.Conv2d(in_channels, n1, kernel_size=f1, padding=f1 // 2)
        self.conv2 = nn.Conv2d(n1, n2, kernel_size=f2, padding=f2 // 2)
        self.conv3 = nn.Conv2d(n2, in_channels - 1, kernel_size=f3, padding=f3 // 2)


        self.relu = nn.ReLU()

    def forward(self, ms: Tensor, pan: Tensor) -> Tensor:
        """
        Args:
            ms (Tensor): [B, C, H, W] multispectral images
            pan (Tensor): [B, 1, H, W] panchromatic image
        Returns:
            Tensor: [B, C-1, H, W] sharpened output
        """
        # concat ms and pan
        x = torch.cat([ms, pan], dim=1)  # [B, C+1, H, W]

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)  # without activation function​​ 

        return x