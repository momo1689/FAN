from torch import nn


class SigmaNet(nn.Module):
    def __init__(self, in_channels, out_channels, depth=3, num_filter=64):
        super(SigmaNet, self).__init__()
        block = []
        block.append(nn.Conv2d(in_channels, num_filter, kernel_size=3, padding=1, bias=True))
        block.append(nn.PReLU())

        for i in range(depth):
            block.append(nn.Conv2d(num_filter, num_filter, kernel_size=3, padding=1, bias=True))
            block.append(nn.PReLU())

        block.append(nn.Conv2d(num_filter, out_channels, kernel_size=3, padding=1, bias=True))
        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out

