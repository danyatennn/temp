import lpips
import torch

from src.metrics.base_metric import BaseMetric
from src.utils.lensless import crop_roi


def _resolve_device(device):
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


class LenslessMetric(BaseMetric):
    """
    Wraps a TorchMetrics metric and applies it on the region of interest.

    Used for PSNR, SSIM, and MSE. Both reconstruction and ground truth are
    cropped to the ROI and clamped to [0, 1].
    """

    def __init__(self, metric, device, *args, **kwargs):
        """
        Args:
            metric (Callable): TorchMetrics metric instance.
            device (str): device for the metric computation.
        """
        super().__init__(*args, **kwargs)
        self.metric = metric.to(_resolve_device(device))

    def __call__(self, reconstruction, gt, **batch):
        rec = crop_roi(reconstruction).clamp(0, 1)
        target = crop_roi(gt).clamp(0, 1)
        return self.metric(rec, target).item()


class LPIPSMetric(BaseMetric):
    """
    LPIPS (VGG) perceptual metric computed on the region of interest.
    """

    def __init__(self, device, net="vgg", *args, **kwargs):
        """
        Args:
            device (str): device for the metric computation.
            net (str): backbone for LPIPS ('vgg' or 'alex').
        """
        super().__init__(*args, **kwargs)
        self.lpips = lpips.LPIPS(net=net).to(_resolve_device(device))
        for param in self.lpips.parameters():
            param.requires_grad = False

    def __call__(self, reconstruction, gt, **batch):
        rec = crop_roi(reconstruction).clamp(0, 1) * 2 - 1
        target = crop_roi(gt).clamp(0, 1) * 2 - 1
        return self.lpips(rec, target).mean().item()
