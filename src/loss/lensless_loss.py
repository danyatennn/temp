import lpips
import torch
from torch import nn

from src.utils.lensless import crop_roi


class MSELPIPSLoss(nn.Module):
    """
    Combined MSE and LPIPS loss computed on the region of interest.

    Matches the training objective of the learned reconstruction papers: a
    pixel-wise MSE term and a perceptual LPIPS (VGG) term. The LPIPS network
    is frozen and not optimized.
    """

    def __init__(self, mse_weight=1.0, lpips_weight=1.0, net="vgg"):
        """
        Args:
            mse_weight (float): weight of the MSE term.
            lpips_weight (float): weight of the LPIPS term.
            net (str): backbone for LPIPS ('vgg' or 'alex').
        """
        super().__init__()
        self.mse_weight = mse_weight
        self.lpips_weight = lpips_weight
        self.mse = nn.MSELoss()
        self.lpips = lpips.LPIPS(net=net)
        for param in self.lpips.parameters():
            param.requires_grad = False

    def forward(self, reconstruction, gt, **batch):
        """
        Args:
            reconstruction (Tensor): predicted reconstruction (B, C, H, W).
            gt (Tensor): ground-truth image (B, C, H, W).
        Returns:
            dict: dict with 'loss', 'mse_loss', and 'lpips_loss' keys.
        """
        rec = crop_roi(reconstruction)
        target = crop_roi(gt)

        mse = self.mse(rec, target)
        lpips_value = self.lpips(
            rec.clamp(0, 1) * 2 - 1, target.clamp(0, 1) * 2 - 1
        ).mean()

        loss = self.mse_weight * mse + self.lpips_weight * lpips_value
        return {"loss": loss, "mse_loss": mse, "lpips_loss": lpips_value}
