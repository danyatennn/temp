import math

import torch
from torch import nn


class FISTA(nn.Module):
    """
    FISTA camera inversion for lensless imaging.

    A non-ADMM solver: accelerated proximal gradient descent on the
    data-fidelity term ``0.5 ||C H x - b||^2`` with an anisotropic
    total-variation prior and a non-negativity constraint. Unlike ADMM there is
    no variable splitting or dual variables; the TV proximal operator is solved
    with a few Chambolle dual iterations. With ``learnable=False`` the gradient
    step is set to ``1 / L`` (the data-term Lipschitz constant); with
    ``learnable=True`` per-iteration step scales and TV weights are optimized.

    The crop operation is handled by working on a ``pad_factor`` times larger
    space with the scene at the center; convolutions are computed via FFT.
    """

    def __init__(
        self,
        n_iters=100,
        learnable=False,
        tau=2e-4,
        n_inner=5,
        pad_factor=2,
    ):
        """
        Args:
            n_iters (int): number of (unrolled) FISTA iterations.
            learnable (bool): if True, step scales and TV weights are trainable.
            tau (float): initial TV sparsity weight.
            n_inner (int): Chambolle iterations for the TV proximal operator.
            pad_factor (int): factor for the padded FFT space size.
        """
        super().__init__()
        self.n_iters = n_iters
        self.n_inner = n_inner
        self.pad_factor = pad_factor

        log_scale = torch.zeros(n_iters)
        log_tau = torch.full((n_iters,), math.log(tau))
        if learnable:
            self.log_scale = nn.Parameter(log_scale)
            self.log_tau = nn.Parameter(log_tau)
        else:
            self.register_buffer("log_scale", log_scale)
            self.register_buffer("log_tau", log_tau)

    def _params(self, k):
        """
        Get the (positive) step scale and TV weight for iteration k.
        """
        return torch.exp(self.log_scale[k]), torch.exp(self.log_tau[k])

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

    def _prox_tv(self, z, weight):
        """
        Proximal operator of the anisotropic total variation.

        Solved with Chambolle dual projected-gradient iterations. ``weight``
        broadcasts over the batch and may be a per-sample tensor.
        """
        p_h = torch.zeros_like(z)
        p_v = torch.zeros_like(z)
        for _ in range(self.n_inner):
            x = z - self._grad_transpose(p_h, p_v)
            dh, dv = self._grad(x)
            p_h = torch.clamp(p_h + dh / 8.0, min=-weight, max=weight)
            p_v = torch.clamp(p_v + dv / 8.0, min=-weight, max=weight)
        return z - self._grad_transpose(p_h, p_v)

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
        lipschitz = (torch.abs(psf_fft) ** 2).amax(dim=(-2, -1), keepdim=True) + 1e-6

        crop_mask = self._pad(torch.ones_like(measurement), full_shape)
        padded_measurement = self._pad(measurement, full_shape)

        x = torch.zeros_like(crop_mask)
        y = torch.zeros_like(crop_mask)
        t = 1.0

        for k in range(self.n_iters):
            scale, tau = self._params(k)
            step = scale / lipschitz

            residual = crop_mask * self._conv(y, psf_fft) - padded_measurement
            grad = self._conv_transpose(residual, psf_fft)
            z = y - step * grad

            x_next = self._prox_tv(z, step * tau)
            x_next = torch.clamp(x_next, min=0.0)

            t_next = (1.0 + math.sqrt(1.0 + 4.0 * t * t)) / 2.0
            y = x_next + ((t - 1.0) / t_next) * (x_next - x)
            x = x_next
            t = t_next

        x = self._crop(x, shape)
        return torch.clamp(x, min=0.0)
