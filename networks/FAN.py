import torch
import torch.nn.functional as func
from torch_wavelet.SWT import SWTForward, SWTInverse
from networks.SigmaNet import SigmaNet
from networks.UNet_enhance import UNet
from networks.net_tool import rgb2yuv, yuv2rgb


class FAN(torch.nn.Module):
    def __init__(self, depth_S=5, depth_U=4, feature_dims=64, wave_pattern='db1', level=1):
        super(FAN, self).__init__()
        self.sigma_net = SigmaNet(in_channels=3, out_channels=3, depth=depth_S, num_filter=feature_dims)
        self.UNet = UNet(in_channels=level*3+6, out_channels=level*3+3, depth=depth_U, feature_dims=feature_dims)
        self.wave_pattern = wave_pattern
        self.level = level

    def forward(self, org_data):
        noise_map = self.sigma_net(org_data)
        noise_map_yuv = rgb2yuv(noise_map)
        if org_data.device == torch.device('cpu'):
            decompose = SWTForward(wave=self.wave_pattern, level=self.level)
            reconstrution = SWTInverse(wave=self.wave_pattern, level=self.level)
        else:
            decompose = SWTForward(wave=self.wave_pattern, level=self.level).cuda()
            reconstrution = SWTInverse(wave=self.wave_pattern, level=self.level).cuda()

        org_yuv = rgb2yuv(org_data)
        img_y = torch.unsqueeze(org_yuv[:, 0], dim=1)
        img_uv = org_yuv[:, 1:]
        out = decompose(img_y)
        cA = out[0]
        high_coeffs = out[1]
        net_input = torch.cat([cA] + high_coeffs + [img_uv] + [noise_map_yuv], dim=1)
        net_out = self.UNet(net_input)

        out_cA = net_out[:, 0].unsqueeze(dim=1)
        out_coeffs = net_out[:, 1:self.level*3+1]
        out_uv = net_out[:, self.level*3+1:]
        out_recons = (out_cA, out_coeffs)
        out_y = reconstrution(out_recons)

        out_yuv = torch.cat([out_y, out_uv], dim=1)
        out_rgb = yuv2rgb(out_yuv)

        return noise_map, out_rgb


if __name__ == '__main__':
    model = FAN()
    print(model)
