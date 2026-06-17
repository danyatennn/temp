import torch
from torch import nn
from torch.nn import functional as F

DEFAULT_WEIGHTS_URL = (
    "https://github.com/xinntao/Real-ESRGAN/releases/download/"
    "v0.1.0/RealESRGAN_x4plus.pth"
)


class ResidualDenseBlock(nn.Module):
    """
    Residual dense block with five convolutions and a residual scaling of 0.2.
    """

    def __init__(self, num_feat=64, num_grow_ch=32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """
    Residual-in-residual dense block: three dense blocks and a residual.
    """

    def __init__(self, num_feat, num_grow_ch=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """
    RRDBNet generator (the Real-ESRGAN architecture).

    Implemented here with the official layer names so that the published
    pretrained weights load directly. The body runs at the input resolution and
    the two upsampling stages give a fixed 4x upscaling.
    """

    def __init__(
        self, num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32
    ):
        super().__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(
            *[RRDB(num_feat, num_grow_ch) for _ in range(num_block)]
        )
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(
            self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest"))
        )
        feat = self.lrelu(
            self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest"))
        )
        return self.conv_last(self.lrelu(self.conv_hr(feat)))


class RealESRGAN(nn.Module):
    """
    Pretrained Real-ESRGAN restorer used as a post-processor.

    The generator upscales its input by 4x; the output is resized back to the
    input resolution so the module acts as a same-resolution image-restoration
    network on top of a reconstruction. Set ``freeze=True`` to use the frozen
    pretrained model, or ``freeze=False`` to fine-tune the generator.
    """

    def __init__(
        self,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        pretrained=True,
        weights_url=DEFAULT_WEIGHTS_URL,
        freeze=False,
    ):
        """
        Args:
            num_feat (int): number of generator features.
            num_block (int): number of RRDB blocks.
            num_grow_ch (int): growth channels of the dense blocks.
            pretrained (bool): if True, load the official pretrained weights.
            weights_url (str): URL of the pretrained weights.
            freeze (bool): if True, freeze the generator parameters.
        """
        super().__init__()
        self.net = RRDBNet(3, 3, num_feat, num_block, num_grow_ch)
        if pretrained:
            self._load_pretrained(weights_url)
        if freeze:
            for param in self.net.parameters():
                param.requires_grad = False

    def _load_pretrained(self, weights_url):
        """
        Download and load the official pretrained generator weights.
        """
        state = torch.hub.load_state_dict_from_url(weights_url, map_location="cpu")
        if "params_ema" in state:
            state = state["params_ema"]
        elif "params" in state:
            state = state["params"]
        self.net.load_state_dict(state, strict=True)

    def forward(self, x):
        """
        Args:
            x (Tensor): reconstruction of shape (B, C, H, W) in [0, 1].
        Returns:
            Tensor: restored image of shape (B, C, H, W).
        """
        h, w = x.shape[-2:]
        y = self.net(x.clamp(0.0, 1.0))
        return F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
