import torch
from torch import nn
import torch.nn.functional as func
from networks.subBlock import SCAB


class ResidualBlock(nn.Module):
    def __init__(self, in_size, out_size, shortcut=None):
        super(ResidualBlock, self).__init__()
        self.basic = nn.Sequential(
            nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True),
            nn.PReLU()
        )
        self.shortcut = shortcut

    def forward(self, x):
        out = self.basic(x)
        residual = x if self.shortcut is None else self.shortcut(x)
        out += residual
        return out


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, num):
        super(UNetConvBlock, self).__init__()
        # print('conv block in_size =', in_size)
        block = []

        block.append(nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True))
        block.append(nn.PReLU())

        for i in range(max(num-1, 1)):
            block.append(ResidualBlock(out_size, out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, num):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.conv_block = UNetConvBlock(in_size, out_size, num)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out


class UNet(nn.Module):
    def __init__(self, in_channels=5, out_channels=4, depth=4, feature_dims=64):
        super(UNet, self).__init__()
        self.depth = depth
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(UNetConvBlock(prev_channels, (2**i)*feature_dims, depth-i))
            if i != depth - 1:
                self.down_path.append(SCAB((2**i)*feature_dims, (2**i)*feature_dims))
            prev_channels = (2**i) * feature_dims

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth-1)):
            self.up_path.append(UNetUpBlock(prev_channels, (2**i)*feature_dims, depth-i))
            prev_channels = (2**i) * feature_dims

        self.last = nn.Conv2d(prev_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if (i != len(self.down_path)-1) and (i % 2 == 1):
                blocks.append(x)
                x = func.avg_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i-1])

        return self.last(x)


if __name__ == '__main__':
    model = UNet()
    print(model)
