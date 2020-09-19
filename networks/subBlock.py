import torch.nn as nn
import torch


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1):
        super(ConvBlock, self).__init__()
        block = []
        block.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=True, dilation=dilation))
        block.append(nn.PReLU())
        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class SpatialAttention(nn.Module):
    """
    Spatial attention part
    """
    def __init__(self, in_channels, reduction=4):
        super(SpatialAttention, self).__init__()
        block = []
        out_stage1 = in_channels // reduction
        out_stage2 = out_stage1 // reduction
        block.append(ConvBlock(in_channels=in_channels, out_channels=out_stage1))
        block.append(ConvBlock(in_channels=out_stage1, out_channels=out_stage2))
        block.append(nn.Conv2d(out_stage2, 1, kernel_size=1, padding=0, bias=True))
        block.append(nn.Sigmoid())
        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class SAB_astrous(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SAB_astrous, self).__init__()
        block = []
        block.append(ConvBlock(in_channels=in_channels, out_channels=in_channels))
        out_stage1 = in_channels // reduction
        block.append(ConvBlock(in_channels=in_channels, out_channels=out_stage1, kernel_size=1, padding=0))
        block.append(ConvBlock(in_channels=out_stage1, out_channels=out_stage1, kernel_size=3, padding=2, dilation=2))
        out_stage2 = out_stage1 // reduction
        block.append(ConvBlock(in_channels=out_stage1, out_channels=out_stage2, kernel_size=1, padding=0))
        block.append(ConvBlock(in_channels=out_stage2, out_channels=out_stage2, kernel_size=3, padding=4, dilation=4))
        block.append(nn.Conv2d(out_stage2, 1, kernel_size=1, padding=0, bias=True))
        block.append(nn.Sigmoid())
        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class ChannelAttention(nn.Module):
    """
    Channel attention part
    """
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        block = []
        block.append(nn.AdaptiveAvgPool2d((1, 1)))
        block.append(ConvBlock(in_channels=channels, out_channels=channels // reduction, kernel_size=1, padding=0))
        block.append(nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=True))
        block.append(nn.Sigmoid())
        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class SCAB(nn.Module):
    """
    Dual attention block
    """
    def __init__(self, org_channels, out_channels):
        super(SCAB, self).__init__()
        pre_x = []
        pre_x.append(ConvBlock(in_channels=org_channels, out_channels=out_channels))
        pre_x.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))

        self.pre_x = nn.Sequential(*pre_x)
        self.CAB = ChannelAttention(channels=out_channels)
        self.SAB = SAB_astrous(in_channels=out_channels)
        self.last = torch.nn.Conv2d(in_channels=2*out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        pre_x = self.pre_x(x)
        # pre_map = self.pre_map(map)
        channel = self.CAB(pre_x)
        spatial = self.SAB(pre_x)
        out_s = pre_x * spatial.expand_as(pre_x)
        out_c = pre_x * channel.expand_as(pre_x)
        out_combine = torch.cat([out_s, out_c], dim=1)
        out = self.last(out_combine)
        out = x + out
        return out
