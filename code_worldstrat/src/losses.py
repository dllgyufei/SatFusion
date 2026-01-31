import itertools
import kornia
import torch
import torch.nn.functional as F
from torch.nn.functional import interpolate
from torch import Tensor, nn
from pytorch_msssim import ms_ssim
from src.transforms import lanczos_kernel
import math
from torch.autograd import Variable
eps = torch.finfo(torch.float32).eps

# 计算PSNR
def psnr_loss(y_hat, y):
    """ Peak Signal to Noise Ratio (PSNR) loss.
    The logarithm of base ten of the mean squared error between the label
    and the output, multiplied by ten.

    In the proper form, there should be a minus sign in front of the equation,
    but since we want to maximize the PSNR,
    we minimize the negative PSNR (loss), thus the leading minus sign has been omitted.

    Parameters
    ----------
    y_hat : Tensor
        Output, tensor of shape (batch_size, channels, height, width).
    y : Tensor
        Label, tensor of shape (batch_size, channels, height, width).

    Returns
    -------
    Tensor
        Tensor of shape (batch_size,).
    """
    return 10.0 * torch.log10(mse_loss(y_hat, y))

# 计算SSIM
def _ssim(img1, img2):
    if img1.ndim == 3:
        img1 = img1.unsqueeze(0)
    if img2.ndim == 3:  # [C, H, W]
        img2 = img2.unsqueeze(0)  # -> [1, C, H, W]
    img1 = img1.float()
    img2 = img2.float()

    channel = img1.shape[1]
    max_val = 1
    _, c, w, h = img1.size()
    window_size = min(w, h, 11)
    sigma = 1.5 * window_size / 11
    window = create_window(window_size, sigma, channel).cuda()
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2
    V1 = 2.0 * sigma12 + C2
    V2 = sigma1_sq + sigma2_sq + C2
    ssim_map = ((2 * mu1_mu2 + C1) * V1) / ((mu1_sq + mu2_sq + C1) * V2)
    t = ssim_map.shape
    return ssim_map.mean(2).mean(2)
def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()
def create_window(window_size, sigma, channel):
    _1D_window = gaussian(window_size, sigma).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

# 计算SSIM_loss
def ssim_loss(y_hat, y, window_size=5):
    """ Structural Similarity loss.
    See: http://www.cns.nyu.edu/pub/eero/wang03-reprint.pdf

    Parameters
    ----------
    y_hat : Tensor
        Output, tensor of shape (batch_size, channels, height, width).
    y : Tensor
        Label, tensor of shape (batch_size, channels, height, width).
    window_size : int, optional
        The size of the gaussian kernel used in the SSIM calculation to smooth the images, by default 5.

    Returns
    -------
    Tensor
        Tensor of shape (batch_size,).
    """
    return kornia.losses.ssim_loss(
        y_hat, y, window_size=window_size, reduction="none"
    ).mean(
        dim=(-1, -2, -3)
    )  # over C, H, W

# 计算MAE
def mae_loss(y_hat, y):
    """ Mean Absolute Error (L1) loss.
    Sum of all the absolute differences between the label and the output.

    Parameters
    ----------
    y_hat : Tensor
        Output, tensor of shape (batch_size, channels, height, width).
    y : Tensor
        Label, tensor of shape (batch_size, channels, height, width).

    Returns
    -------
    Tensor
        Tensor of shape (batch_size,).
    """
    return F.l1_loss(y_hat, y, reduction="none").mean(dim=(-1, -2, -3))  # over C, H, W

# 计算MSE
def mse_loss(y_hat, y):
    """ Mean Squared Error (L2) loss.
    Sum of all the squared differences between the label and the output.

    Parameters
    ----------
    y_hat : Tensor
        Output, tensor of shape (batch_size, channels, height, width).
    y : Tensor
        Label, tensor of shape (batch_size, channels, height, width).

    Returns
    -------
    Tensor
        Tensor of shape (batch_size,).
    """
    return F.mse_loss(y_hat, y, reduction="none").mean(dim=(-1, -2, -3))  # over C, H, W

# 计算SAM
def _sam(y_hat: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Spectral Angle Mapper (SAM).

    Parameters
    ----------
    y_hat : Tensor
        Output, shape (B, C, H, W).
    y : Tensor
        Label, shape (B, C, H, W).

    Returns
    -------
    Tensor
        SAM values per sample in degrees, shape (B,).
    """
    B, C, H, W = y.shape
    y_hat = y_hat.permute(0, 2, 3, 1).reshape(B, -1, C)  # (B, HW, C)
    y = y.permute(0, 2, 3, 1).reshape(B, -1, C)          # (B, HW, C)

    dot_product = (y_hat * y).sum(dim=-1)                # (B, HW)
    norm1 = y_hat.norm(p=2, dim=-1)                      # (B, HW)
    norm2 = y.norm(p=2, dim=-1)                          # (B, HW)
    denom = norm1 * norm2 + eps

    cos_theta = (dot_product / denom).clamp(-1+eps, 1-eps)
    angle = torch.acos(cos_theta)                        # (B, HW)
    sam_val = torch.mean(angle, dim=1) * 180 / 3.14159265  # (B,)
    return sam_val

# 计算SAM_loss
def sam_loss(y_hat, y, eps=1e-8):
    """SAM for (B, C, H, W) image; torch.float32 [0.,1.]."""
    inner_product = (y_hat * y).sum(dim=1)
    img1_spectral_norm = torch.sqrt((y_hat**2).sum(dim=1))
    img2_spectral_norm = torch.sqrt((y**2).sum(dim=1))
    cos_theta = (inner_product / (img1_spectral_norm * img2_spectral_norm + eps)).clamp(min=-1 + eps, max=1 - eps)
    loss = 1 - cos_theta
    loss = loss.reshape(loss.shape[0], -1)
    return loss.mean(dim=-1).mean()

# 计算ERGAS(www_code)
def ergas(y_hat: torch.Tensor, y: torch.Tensor, ratio: int = 4, eps: float = 1e-8) -> torch.Tensor:
    """
    Relative Dimensionless Global Error in Synthesis (ERGAS).

    Parameters
    ----------
    y_hat : Tensor
        Output, shape (B, C, H, W).
    y : Tensor
        Label, shape (B, C, H, W).
    ratio : int
        Resolution ratio between PAN and MS (default=4).
    eps : float
        Small value to avoid division by zero.

    Returns
    -------
    Tensor
        ERGAS values per sample, shape (B,).
    """
    B, C, H, W = y.shape
    # (B, C) 每个样本、每个通道的 MSE
    mse = ((y_hat - y) ** 2).mean(dim=(2, 3))  # (B, C)

    # (B, C) 每个样本、每个通道的均值平方
    mean_sq = (y.mean(dim=(2, 3)) ** 2) + eps  # (B, C) #NIPS

    # channel_means = torch.tensor([0.1667,0.0518,0.0659], device=y.device)  # shape (C,)
    # mean_sq = (channel_means ** 2 + eps).unsqueeze(0).expand(B, -1)  # (B, C) #WWW,用“数据集级”通道均值做分母（稳定且标准做法）,ERGAS不再随样本亮度漂移。

    # sum over channels
    summ = (mse / mean_sq).sum(dim=1) / C  # (B,)
    ergas_val = 100.0 * (1.0 / ratio) * torch.sqrt(summ)  # (B,)

    return ergas_val

# # 计算ERGAS(nips_code)
# def ergas(y_hat, y, scale=4):
#     """ERGAS for (N, C, H, W) image; torch.float32 [0.,1.].
#     scale = spatial resolution of PAN / spatial resolution of MUL, default 4."""
#
#     B, C, H, W = y.shape
#     means_real = y.reshape(B, C, -1).mean(dim=-1)
#     mses = ((y_hat - y) ** 2).reshape(B, C, -1).mean(dim=-1)
#     # Warning: There is a small value in the denominator for numerical stability.
#     # Since the default dtype of torch is float32, our result may be slightly different from matlab or numpy based ERGAS
#
#     return 100 / scale * torch.sqrt((mses / (means_real ** 2 + eps)).mean())

# 计算多尺度结构相似性损失，用于衡量两幅图像的相似程度，MSE 只衡量像素级别的误差，而 MS-SSIM 更关注图像的结构信息，MS-SSIM 越大表示两张图像越相似
def ms_ssim_loss(y_hat, y, window_size):
    """ Multi-Scale Structural Similarity loss.
    See: https://www.cns.nyu.edu/pub/eero/wang03b.pdf

    Parameters
    ----------
    y_hat : Tensor
        Output, tensor of shape (batch_size, channels, height, width).
    y : Tensor
        Label, tensor of shape (batch_size, channels, height, width).
    window_size : int
        The size of the gaussian kernel used in the MS-SSIM calculation to smooth the images.

    Returns
    -------
    Tensor
        Tensor of shape (batch_size,).
    """
    return 1 - ms_ssim(y_hat, y, data_range=1, win_size=window_size, size_average=False)     # 损失函数是希望更小越好，因此用 1 - MS-SSIM 作为 Loss

def get_patch(x, x_start, y_start, patch_size):
    """ Get a patch of the input tensor. 
    The patch begins at the coordinates (x_start, y_start) ,
    ends at (x_start + patch_size, y_start + patch_size).

    Parameters
    ----------
    x : Tensor
        Tensor of shape (batch_size, channels, height, width).
    x_start : int
        The x-coordinate of the top left corner of the patch.
    y_start : int
        The y-coordinate of the top left corner of the patch.
    patch_size : int
        The height/width of the (square) patch.

    Returns
    -------
    Tensor
        Tensor of shape (batch_size, channels, patch_size, patch_size).
    """
    return x[..., x_start : (x_start + patch_size), y_start : (y_start + patch_size)]


class Shift(nn.Module):
    """ A non-learnable convolutional layer for shifting. 
    Used instead of ShiftNet.
    """

    def __init__(self, shift_by_px, mode="discrete", step=1.0, use_cache=True):
        """ Initialize the Shift layer.

        Parameters
        ----------
        shift_by_px : int
            The number of pixels to shift the input by.
        mode : str, optional
            The mode of shifting, by default 'discrete'.
        step : float, optional
            The step size of the shift, by default 1.0.
        use_cache : bool, optional
            Whether to cache the shifts, by default True.
        """
        super().__init__()
        self.shift_by_px = shift_by_px
        self.mode = mode
        if mode == "discrete":
            shift_kernels = self._shift_kernels(shift_by_px)
        elif mode == "lanczos":
            shift_kernels = self._lanczos_kernels(shift_by_px, step)
        self._make_shift_conv2d(shift_kernels)
        self.register_buffer("shift_kernels", shift_kernels)
        self.y = None
        self.y_hat = None
        self.use_cache = use_cache
        self.shift_cache = {}

    def _make_shift_conv2d(self, kernels):
        """ Make the shift convolutional layer.

        Parameters
        ----------
        kernels : torch.Tensor
            The shift kernels.
        """
        self.number_of_kernels, _, self.kernel_height, self.kernel_width = kernels.shape
        self.conv2d_shift = nn.Conv2d(
            in_channels=self.number_of_kernels,
            out_channels=self.number_of_kernels,
            kernel_size=(self.kernel_height, self.kernel_width),
            bias=False,
            groups=self.number_of_kernels,
            padding_mode="reflect",
        )

        # Fix (kN, 1, kH, kW)
        self.conv2d_shift.weight.data = kernels
        self.conv2d_shift.requires_grad_(False)  # Freeze

    @staticmethod
    def _shift_kernels(shift_by_px):
        """ Create the shift kernels.

        Parameters
        ----------
        shift_by_px : int
            The number of pixels to shift the input by.

        Returns
        -------
        torch.Tensor
            The shift kernels.
        """
        kernel_height = kernel_width = (2 * shift_by_px) + 1
        kernels = torch.zeros(
            kernel_height * kernel_width, 1, kernel_height, kernel_width
        )
        all_xy_positions = list(
            itertools.product(range(kernel_height), range(kernel_width))
        )

        for kernel, (x, y) in enumerate(all_xy_positions):
            kernels[kernel, 0, x, y] = 1
        return kernels

    @staticmethod
    def _lanczos_kernels(shift_by_px, shift_step):
        """ Create the Lanczos kernels.

        Parameters
        ----------
        shift_by_px : int
            The number of pixels to shift the input by.
        shift_step : float
            The step size of the shift.

        Returns
        -------
        torch.Tensor
            The Lanczos kernels.
        """
        shift_step = float(shift_step)
        shifts = torch.arange(-shift_by_px, shift_by_px + shift_step, shift_step)
        shifts = shifts[:, None]
        kernel = lanczos_kernel(shifts, kernel_lobes=3)
        kernels = torch.stack(
            [
                kernel_y[:, None] @ kernel_x[None, :]
                for kernel_y, kernel_x in itertools.product(kernel, kernel)
            ]
        )
        return kernels[:, None]

    def forward(self, y: Tensor) -> Tensor:
        """ Forward shift pass.

        Parameters
        ----------
        y : torch.Tensor
            The input tensor.

        Returns
        -------
        Tensor
            The shifted tensor.
        """
        batch_size, input_channels, input_height, input_width = y.shape
        patch_height = patch_width = y.shape[-1] - self.kernel_width + 1

        number_of_kernels_dimension = -3

        # TODO: explain what is going on here
        y = y.unsqueeze(dim=number_of_kernels_dimension).expand(
            -1, -1, self.number_of_kernels, -1, -1
        )  # (B, C, kN, H, W)

        y = y.view(
            batch_size * input_channels,
            self.number_of_kernels,
            input_height,
            input_width,
        )

        # Current input shape: (number_of_kernels, batch_channels, height, width)
        y = self.conv2d_shift(y)
        batch_size_channels, number_of_kernels, height, width = 0, 1, 2, 3

        # Transposed input shape: (number_of_kernels, batch_size_channels, height, width)
        y = y.transpose(number_of_kernels, batch_size_channels)

        y = y.contiguous().view(
            self.number_of_kernels * batch_size,
            input_channels,
            patch_height,
            patch_width,
        )
        return y

    def prep_y_hat(self, y_hat):
        """ Prepare the y_hat for the shift by

        Parameters
        ----------
        y_hat : torch.Tensor
            The output tensor.

        Returns
        -------
        Tensor
            ???
        """
        patch_width = y_hat.shape[-1] - self.kernel_width + 1

        # Get center patch
        y_hat = get_patch(
            y_hat,
            x_start=self.shift_by_px,
            y_start=self.shift_by_px,
            patch_size=patch_width,
        )

        # (number_of_kernels, batch_size, channels, height, width)
        y_hat = y_hat.expand(self.number_of_kernels, -1, -1, -1, -1)
        _, batch_size, channels, height, width = y_hat.shape

        # (number_of_kernels*batch_size, channels, height, width)
        return y_hat.contiguous().view(
            self.number_of_kernels * batch_size, channels, height, width
        )

    @staticmethod
    def gather_shifted_y(y: Tensor, ix) -> Tensor:
        """ Gather the shifted y.

        Parameters
        ----------
        y : Tensor
            The input tensor.
        ix : Tensor
            ???

        Returns
        -------
        Tensor
            The shifted y.
        """
        batch_size = ix.shape[0]
        # TODO: Check if 1st dimension is number of kernels
        number_of_kernels_batch_size, channels, height, width = y.shape
        number_of_kernels = number_of_kernels_batch_size // batch_size
        ix = ix[None, :, None, None, None].expand(-1, -1, channels, height, width)

        # (batch_size, channels, height, width)
        return y.view(number_of_kernels, batch_size, channels, height, width).gather(
            dim=0, index=ix
        )[0]

    @staticmethod
    def _hash_y(y):
        """ Hashes y by [???].

        Parameters
        ----------
        y : Tensor
            The input tensor.

        Returns
        -------
        Tensor
            The hashed tensor.
        """
        batch_size = y.shape[0]
        return [tuple(row.tolist()) for row in y[:, :4, :4, :4].reshape(batch_size, -1)]

    def registered_loss(self, loss_function):
        """ Creates a loss function adjusted for registration errors 
        by computing the min loss across shifts of up to `self.shift_by_px` pixels.

        Parameters
        ----------
        loss_function : Callable
            The loss function.

        Returns
        -------
        Callable
            The loss function adjusted for registration errors.

        """

        def _loss(y_hat, y=None, **kws):
            """ Compute the loss.

            Parameters
            ----------
            y_hat : Tensor
                The output tensor.
            y : Tensor, optional
                The target tensor, by default None.

            Returns
            -------
            Tensor
                The loss.
            """

            hashed_y = self._hash_y(y)
            cached_y = (
                torch.Tensor([hash in self.shift_cache for hash in hashed_y])
                .bool()
                .to(y.device)
            )
            not_cached_y = ~cached_y

            # If y and y_hat are both cached, return the loss
            if self.y is not None and self.y_hat is not None:
                min_loss = loss_function(self.y_hat, self.y, **kws)
            else:
                batch_size, channels, height, width = y.shape
                min_loss = torch.zeros(batch_size).to(y.device)
                patch_width = width - self.kernel_width + 1  # patch width

                y_all = torch.zeros(batch_size, channels, patch_width, patch_width).to(
                    y.device
                )

                y_hat_all = torch.zeros(
                    batch_size, channels, patch_width, patch_width
                ).to(y.device)

                # If there are any hashes in cache
                if any(cached_y):

                    ix = torch.stack(
                        [
                            self.shift_cache[hash]
                            for hash in hashed_y
                            if hash in self.shift_cache
                        ]
                    )

                    optimal_shift_kernel = self.shift_kernels[ix]
                    print(optimal_shift_kernel.shape)
                    (
                        batch_size,
                        number_of_kernels,
                        _,
                        kernel_height,
                        kernel_width,
                    ) = optimal_shift_kernel.shape

                    conv2d_shift = nn.Conv2d(
                        in_channels=number_of_kernels,
                        out_channels=number_of_kernels,
                        kernel_size=(kernel_height, kernel_width),
                        bias=False,
                        groups=number_of_kernels,
                        padding_mode="reflect",
                    )

                    # Fix and freeze (kN, 1, kH, kW)
                    conv2d_shift.weight.data = optimal_shift_kernel
                    conv2d_shift.requires_grad_(False)
                    y_in = conv2d_shift(y[cached_y].transpose(-3, -4)).transpose(-4, -3)

                    y_hat_in = get_patch(
                        y_hat[cached_y],
                        x_start=self.shift_by_px,
                        y_start=self.shift_by_px,
                        patch_size=patch_width,
                    )  # center patch

                    min_loss[cached_y] = loss_function(y_hat_in, y_in, **kws)
                    y_all[cached_y] = y_in.to(y_all.dtype)
                    y_hat_all[cached_y] = y_hat_in.to(y_hat_all.dtype)

                # If there are any hashes not in cache
                if any(not_cached_y):
                    y_out, y_hat_out = y[not_cached_y], y_hat[not_cached_y]
                    batch_size = y_out.shape[0]
                    y_out = self(y_out)  # (Nbatch, channels, height, width)
                    # (Nbatch, channels, height, width)
                    y_hat_out = self.prep_y_hat(y_hat_out)
                    losses = loss_function(y_hat_out, y_out, **kws).view(
                        -1, batch_size
                    )  # (N, B)
                    min_loss[not_cached_y], ix = torch.min(
                        losses, dim=0
                    )  # min over patches (B,)
                    y_out = self.gather_shifted_y(
                        y_out, ix
                    )  # shifted y (batch_size, channels, height, width)
                    batch_size, channels, height, width = y_out.shape
                    # (batch_size, channels, height, width). Copied along dim 0
                    y_hat_out = y_hat_out.view(-1, batch_size, channels, height, width)

                    y_hat_out = y_hat_out[0]

                    y_all[not_cached_y] = y_out.to(y_all.dtype)
                    y_hat_all[not_cached_y] = y_hat_out.to(y_hat_all.dtype)
                    if self.use_cache:
                        hashed_y = [
                            hash for hash in hashed_y if hash not in self.shift_cache
                        ]
                        for hash, index in zip(hashed_y, ix):
                            self.shift_cache[hash] = ix

                self.y, self.y_hat = y_all, y_hat_all

            return min_loss

        return _loss
