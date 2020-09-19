import torch


def rgb2yuv(img):
    y = 0.299 * img[:, 0] + 0.587 * img[:, 1] + 0.114 * img[:, 2]
    u = -0.169 * img[:, 0] - 0.331 * img[:, 1] + 0.5 * img[:, 2] + 0.5
    v = 0.5 * img[:, 0] - 0.419 * img[:, 1] - 0.081 * img[:, 2] + 0.5
    out = torch.stack((y, u, v))
    out = out.transpose(0, 1)
    return out


def yuv2rgb(img):
    r = img[:, 0] + 1.4075 * (img[:, 2] - 0.5)
    g = img[:, 0] - 0.3455 * (img[:, 1] - 0.5) - 0.7169 * (img[:, 2] - 0.5)
    b = img[:, 0] + 1.779 * (img[:, 1] - 0.5)
    out = torch.stack((r, g, b))
    out = out.transpose(0, 1)
    return out
