import torch
from torch import nn


def soft_threshold(x, thresh):
    """
    Anisotropic (element-wise) soft-thresholding operator.

    Args:
        x (Tensor): input tensor.
        thresh (float | Tensor): threshold value.
    Returns:
        Tensor: thresholded tensor.
    """
    return torch.sign(x) * torch.clamp(torch.abs(x) - thresh, min=0.0)


class UnrolledADMM(nn.Module):
    """
    Unrolled ADMM camera inversion for lensless imaging.

    Implements the variable-splitting ADMM from Monakhova et al. (Le-ADMM).
    With ``learnable=False`` and fixed hyperparameters it reduces to the
    classic ADMM algorithm. With ``learnable=True`` the per-iteration
    penalty parameters and the sparsity threshold are optimized end-to-end.

    The crop operation is handled by working on a ``pad_factor`` times larger
    space with the scene at the center. Convolutions with the PSF and the
    anisotropic total-variation operator are computed in the Fourier domain.
    """

    def __init__(
        self,
        n_iters=5,
        learnable=True,
        mu1=1e-4,
        mu2=1e-4,
        mu3=1e-4,
        tau=2e-4,
        pad_factor=2,
    ):
        """
        Args:
            n_iters (int): number of (unrolled) ADMM iterations.
            learnable (bool): if True, hyperparameters are trainable.
            mu1 (float): initial penalty parameter for the v split (data).
            mu2 (float): initial penalty parameter for the u split (TV).
            mu3 (float): initial penalty parameter for the w split (non-neg).
            tau (float): initial TV sparsity weight.
            pad_factor (int): factor for the padded FFT space size.
        """
        super().__init__()
        self.n_iters = n_iters
        self.pad_factor = pad_factor

        init = torch.log(torch.tensor([mu1, mu2, mu3, tau], dtype=torch.float32))
        log_params = init.unsqueeze(0).repeat(n_iters, 1)
        if learnable:
            self.log_params = nn.Parameter(log_params)
        else:
            self.register_buffer("log_params", log_params)

    def _params(self, k):
        """
        Get the (positive) hyperparameters for iteration k.
        """
        p = torch.exp(self.log_params[k])
        return p[0], p[1], p[2], p[3]

    def _pad(self, x, full_shape):
        """
        Pad an image to the full (padded) shape, placing it at the center.
        """
        h, w = x.shape[-2:]
        full_h, full_w = full_shape
        top = (full_h - h) // 2
        bottom = full_h - h - top
        left = (full_w - w) // 2
        right = full_w - w - left
        return nn.functional.pad(x, (left, right, top, bottom))

    def _crop(self, x, shape):
        """
        Crop the central region of the given shape from a padded image.
        """
        h, w = shape
        full_h, full_w = x.shape[-2:]
        top = (full_h - h) // 2
        left = (full_w - w) // 2
        return x[..., top : top + h, left : left + w]

    @staticmethod
    def _grad(x):
        """
        Circular forward finite differences along height and width.
        """
        dh = torch.roll(x, shifts=1, dims=-2) - x
        dv = torch.roll(x, shifts=1, dims=-1) - x
        return dh, dv

    @staticmethod
    def _grad_transpose(dh, dv):
        """
        Adjoint of the finite-difference operator.
        """
        return (torch.roll(dh, shifts=-1, dims=-2) - dh) + (
            torch.roll(dv, shifts=-1, dims=-1) - dv
        )

    @staticmethod
    def _conv(x, psf_fft):
        """
        Circular convolution with the PSF in the Fourier domain.
        """
        return torch.fft.ifft2(psf_fft * torch.fft.fft2(x)).real

    @staticmethod
    def _conv_transpose(x, psf_fft):
        """
        Adjoint convolution with the PSF in the Fourier domain.
        """
        return torch.fft.ifft2(torch.conj(psf_fft) * torch.fft.fft2(x)).real

    def forward(self, measurement, psf):
        """
        Reconstruct the scene from a lensless measurement.

        Args:
            measurement (Tensor): lensless measurement of shape (B, C, H, W).
            psf (Tensor): per-sample point spread function of shape (B, C, H, W).
        Returns:
            Tensor: reconstruction of shape (B, C, H, W).
        """
        shape = measurement.shape[-2:]
        full_shape = (shape[0] * self.pad_factor, shape[1] * self.pad_factor)

        psf_full = self._pad(psf, full_shape)
        psf_fft = torch.fft.fft2(torch.fft.ifftshift(psf_full, dim=(-2, -1)))
        psf_norm = torch.abs(psf_fft) ** 2

        impulse = torch.zeros(
            (1, 1) + full_shape, device=measurement.device, dtype=measurement.dtype
        )
        impulse[..., 0, 0] = 1.0
        kh, kv = self._grad(impulse)
        tv_norm = (
            torch.abs(torch.fft.fft2(kh)) ** 2 + torch.abs(torch.fft.fft2(kv)) ** 2
        )

        crop_mask = self._pad(torch.ones_like(measurement), full_shape)
        padded_measurement = self._pad(measurement, full_shape)

        x = torch.zeros_like(crop_mask)
        u_h = torch.zeros_like(x)
        u_v = torch.zeros_like(x)
        v = torch.zeros_like(x)
        w = torch.zeros_like(x)
        alpha1 = torch.zeros_like(x)
        alpha2_h = torch.zeros_like(x)
        alpha2_v = torch.zeros_like(x)
        alpha3 = torch.zeros_like(x)

        for k in range(self.n_iters):
            mu1, mu2, mu3, tau = self._params(k)

            dh, dv = self._grad(x)
            u_h = soft_threshold(dh + alpha2_h / mu2, tau / mu2)
            u_v = soft_threshold(dv + alpha2_v / mu2, tau / mu2)

            conv_x = self._conv(x, psf_fft)
            v = (alpha1 + mu1 * conv_x + padded_measurement) / (crop_mask + mu1)

            w = torch.clamp(alpha3 / mu3 + x, min=0.0)

            residual = (
                (mu3 * w - alpha3)
                + self._grad_transpose(mu2 * u_h - alpha2_h, mu2 * u_v - alpha2_v)
                + self._conv_transpose(mu1 * v - alpha1, psf_fft)
            )
            x = torch.fft.ifft2(
                torch.fft.fft2(residual) / (mu1 * psf_norm + mu2 * tv_norm + mu3)
            ).real

            conv_x = self._conv(x, psf_fft)
            alpha1 = alpha1 + mu1 * (conv_x - v)
            dh, dv = self._grad(x)
            alpha2_h = alpha2_h + mu2 * (dh - u_h)
            alpha2_v = alpha2_v + mu2 * (dv - u_v)
            alpha3 = alpha3 + mu3 * (x - w)

        x = self._crop(x, shape)
        return torch.clamp(x, min=0.0)
