
import torch
import torch.nn as nn
import torch.optim as optim
from .model_trnet_pan_dicnn import TRNet_PAN_DiCNN
import numpy as np
import kornia
import torch
import torch.nn.functional as F

# --------------------添加多种损失函数----------------------------------#
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

class SatFusionLoss(nn.Module):
    def __init__(self, w_mse=1.0, w_mae=0.0, w_ssim=0.0, w_sam=0.0):
        super().__init__()
        self.w_mse = w_mse
        self.w_mae = w_mae
        self.w_ssim = w_ssim
        self.w_sam = w_sam

    def forward(self, y_hat, y):
        loss = torch.zeros((), device=y_hat.device)

        if self.w_mse > 0:
            loss = loss + self.w_mse * F.mse_loss(y_hat, y, reduction="mean")

        if self.w_mae > 0:
            loss = loss + self.w_mae * F.l1_loss(y_hat, y, reduction="mean")

        if self.w_ssim > 0:
            loss = loss + self.w_ssim * kornia.losses.ssim_loss(
                y_hat, y, window_size=5, reduction='mean'
            )

        if self.w_sam > 0:
            loss = loss + self.w_sam * sam_loss(y_hat, y)

        return loss
# --------------------添加多种损失函数----------------------------------#

class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, losses, weight_dict):
        """ Create the criterion.
        Parameters:
            num_classes: n able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relatiumber of object categories, omitting the special no-object category
            matcher: moduleve classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.weight_dict = weight_dict
        self.losses = losses
        self.loss_dicts = {}

    def forward(self, outputs, targets, *args, **kwargs):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Compute all the requested losses

        for k in self.losses.keys():
            # k, loss = loss_dict
            if k == 'Loss':
                loss = self.losses[k]
                loss_dicts = loss(outputs, targets)
                if isinstance(loss_dicts, dict):
                    self.loss_dicts.update(loss(outputs, targets))
                else:
                    self.loss_dicts.update({k: loss(outputs, targets)})
            else:
                loss = self.losses[k]
                loss_dicts = loss(outputs, targets, *args)
                if isinstance(loss_dicts, dict):
                    self.loss_dicts.update(loss(outputs, targets, *args))
                else:
                    self.loss_dicts.update({k: loss(outputs, targets, *args)})

        return self.loss_dicts

from UDL.pansharpening.models import PanSharpeningModel
class build_trnet_dicnn(PanSharpeningModel, name='TRNet_PAN_DiCNN1'):
    def __call__(self, cfg):

        # important for Pansharpening models, which are from tensorflow code
        self.reg = cfg.reg

        scheduler = None

        if any(["wv" in v for v in cfg.dataset.values()]):
            spectral_num = 8
        else:
            spectral_num = 4


        # loss = nn.MSELoss(reduction = 'mean').cuda()
        # weight_dict = {'loss': 1}
        # losses = {'loss': loss}

        # --------------------添加多种损失函数----------------------------------#
        loss = SatFusionLoss(
            w_mse=0.3,
            w_mae=0.3,
            w_ssim=0.2,
            w_sam=0.2,
        ).cuda()
        weight_dict = {'loss': 1}
        losses = {'loss': loss}
        # --------------------添加多种损失函数----------------------------------#

        criterion = SetCriterion(losses, weight_dict)
        model = TRNet_PAN_DiCNN(spectral_num, criterion).cuda()
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=0)  ## optimizer 1: Adam
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=1500,
                                                       gamma=0.5)  # lr = lr* gamma for each step_size = 180

        return model, criterion, optimizer, scheduler


###################################################################
# ------------------- Main Function (Run first) -------------------
###################################################################
