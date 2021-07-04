import torch
from torch import nn as nn

from project.models.normaliser import get_use_norm, get_normaliser


class MaskedConv2d(nn.Conv2d):

    def __init__(self, mask_type, in_dim, in_channels, out_channels, kernel_size, stride=1, padding_mode="zeros",
                 dilation=1, groups=1, bias=True):
        padding = (kernel_size - 1) // 2  # keep spatial dimensions
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                         padding_mode=padding_mode, dilation=dilation, groups=groups, bias=bias)

        self.register_buffer("mask", torch.ones_like(self.weight.data))

        centre = kernel_size // 2
        self.mask[:, :, centre, centre + 1:] = 0
        self.mask[:, :, centre + 1:, :] = 0

        for i in range(in_dim):
            for j in range(in_dim):
                if (i >= j and mask_type == "A") or (i > j and mask_type == "B"):
                    self.mask[j::in_dim, i::in_dim, centre, centre] = 0

    def forward(self, x):
        self.weight.data.mul_(self.mask)
        return super().forward(x)


class BasicResidualBlock(nn.Module):
    def __init__(self, in_dim, num_fmaps, h=None, w=None, normaliser=None):
        super().__init__()

        use_norm = get_use_norm(normaliser)
        if use_norm and (h is None or w is None):
            raise ValueError("h and w must be specified when using a normaliser. ")

        # BN-ReLU-Weight
        self.downsample = nn.Sequential(
            get_normaliser(normaliser, 2 * num_fmaps, h, w),
            nn.ReLU(),
            MaskedConv2d("B", in_dim, in_channels=2 * num_fmaps, out_channels=num_fmaps, kernel_size=1,
                         bias=not use_norm)
        )

        self.conv3x3 = nn.Sequential(
            get_normaliser(normaliser, num_fmaps, h, w),
            nn.ReLU(),
            MaskedConv2d("B", in_dim, in_channels=num_fmaps, out_channels=num_fmaps, kernel_size=3, bias=not use_norm)
        )

        self.upsample = nn.Sequential(
            get_normaliser(normaliser, num_fmaps, h, w),
            nn.ReLU(),
            MaskedConv2d("B", in_dim, in_channels=num_fmaps, out_channels=2 * num_fmaps, kernel_size=1,
                         bias=not use_norm)
        )

    def forward(self, x):
        downsample_out = self.downsample(x)
        conv3x3_out = self.conv3x3(downsample_out)
        upsample_out = self.upsample(conv3x3_out)
        return upsample_out + x


class PixelCNN(nn.Module):
    def __init__(self, in_dim, out_dim, num_fmaps, num_blocks, h=None, w=None, normaliser=None):
        super().__init__()

        use_norm = get_use_norm(normaliser)
        if use_norm and (h is None or w is None):
            raise ValueError("h and w must be specified when using a normaliser. ")

        self.conv7x7 = MaskedConv2d("A", in_dim, in_channels=in_dim, out_channels=2 * num_fmaps, kernel_size=7,
                                    bias=not use_norm)

        self.res_blocks = nn.Sequential(
            *[BasicResidualBlock(in_dim, num_fmaps, h, w, normaliser) for _ in range(num_blocks)])
        self.conv1x1 = nn.Sequential(
            get_normaliser(normaliser, 2 * num_fmaps, h, w),
            nn.ReLU(),
            MaskedConv2d("B", in_dim, in_channels=2 * num_fmaps, out_channels=num_fmaps, kernel_size=1,
                         bias=not use_norm),
            get_normaliser(normaliser, num_fmaps, h, w),
            nn.ReLU(),
            MaskedConv2d("B", in_dim, in_channels=num_fmaps, out_channels=out_dim, kernel_size=1)
        )

    def forward(self, x):
        conv7x7_out = self.conv7x7(x)
        res_out = self.res_blocks(conv7x7_out)
        conv1x1_out = self.conv1x1(res_out)
        return conv1x1_out
