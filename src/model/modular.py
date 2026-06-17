import torch
from torch import nn


class ModularReconstruction(nn.Module):
    """
    Modular lensless reconstruction pipeline.

    The pipeline is composed of an optional pre-processor, a camera inversion
    module (unrolled ADMM), and an optional post-processor. Setting both
    processors to None recovers a pure (learned or classic) ADMM reconstruction.
    """

    def __init__(
        self,
        camera_inversion,
        pre_processor=None,
        post_processor=None,
        normalize=True,
    ):
        """
        Args:
            camera_inversion (nn.Module): camera inversion module.
            pre_processor (nn.Module | None): processor applied to the measurement.
            post_processor (nn.Module | None): processor applied to the inversion.
            normalize (bool): if True, scale the inversion output to [0, 1].
        """
        super().__init__()
        self.pre_processor = pre_processor
        self.camera_inversion = camera_inversion
        self.post_processor = post_processor
        self.normalize = normalize

    @staticmethod
    def _normalize(x):
        """
        Scale each image in the batch by its maximum value.
        """
        max_val = torch.amax(x, dim=(-3, -2, -1), keepdim=True)
        return x / (max_val + 1e-6)

    def forward(self, measurement, psf, **batch):
        """
        Args:
            measurement (Tensor): lensless measurement of shape (B, C, H, W).
            psf (Tensor): per-sample PSF of shape (B, C, H, W).
        Returns:
            dict: dict with the reconstruction under the 'reconstruction' key.
        """
        x = measurement
        if self.pre_processor is not None:
            x = self.pre_processor(x)

        x = self.camera_inversion(x, psf)

        if self.normalize:
            x = self._normalize(x)

        if self.post_processor is not None:
            x = self.post_processor(x)

        return {"reconstruction": x}

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum(p.numel() for p in self.parameters())
        trainable_parameters = sum(
            p.numel() for p in self.parameters() if p.requires_grad
        )
        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"
        return result_info
