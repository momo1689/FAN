import numpy as np
import torch
import torch.nn.functional as F


def wave_pad(x, padding, mode='periodic', value=0):
    if mode == 'periodic':
        # only vertical
        if padding[0] == 0 and padding[1] == 0:
            x_pad = np.arange(x.shape[-2])
            x_pad = np.pad(x_pad, (padding[2], padding[3]), mode='wrap')
            return x[:, :, x_pad, :]
        # only horizontal
        elif padding[2] == 0 and padding[3] == 0:
            x_pad = np.arange(x.shape[-1])
            x_pad = np.pad(x_pad, (padding[0], padding[1]), mode='wrap')
            return x[:, :, :, x_pad]
        # both
        else:
            x_pad_col = np.arange(x.shape[-2])
            x_pad_col = np.pad(x_pad_col, (padding[2], padding[3]), mode='wrap')
            x_pad_row = np.arange(x.shape[-1])
            x_pad_row = np.pad(x_pad_row, (padding[0], padding[1]), mode='wrap')
            col = np.outer(x_pad_col, np.ones(x_pad_row.shape[0]))
            row = np.outer(np.ones(x_pad_col.shape[0]), x_pad_row)
            return x[:, :, col, row]
    elif mode == 'constant' or mode == 'reflect' or mode == 'replicate':
        return F.pad(x, padding, mode, value)
    else:
        raise ValueError("Unknown padding mode".format(mode))


def prep_filt(low_col, high_col, low_row=None, high_row=None):
    low_col = np.array(low_col).ravel()
    high_col = np.array(high_col).ravel()
    if low_row is None:
        low_row = low_col
    else:
        low_row = np.array(low_row).ravel()
    if high_row is None:
        high_row = high_col
    else:
        high_row = np.array(high_row).ravel()
    low_col = torch.from_numpy(low_col).reshape((1, 1, -1, 1)).float()
    high_col = torch.from_numpy(high_col).reshape((1, 1, -1, 1)).float()
    low_row = torch.from_numpy(low_row).reshape((1, 1, 1, -1)).float()
    high_row = torch.from_numpy(high_row).reshape((1, 1, 1, -1)).float()

    return low_col, high_col, low_row, high_row


def upsample(filts):
    new_filts = []
    for f in filts:
        if f.shape[3] == 1:
            new = torch.zeros((f.shape[0], f.shape[1], 2*f.shape[2], f.shape[3]), dtype=torch.float, device=f.device)
            new[:, :, ::2, :] = f.clone()
        else:
            new = torch.zeros((f.shape[0], f.shape[1], f.shape[2], 2*f.shape[3]), dtype=torch.float, device=f.device)
            new[:, :, :, ::2] = f.clone()
        new_filts.append(new)
    return new_filts


def afb1d(x_pad, low, high, dim):
    """
    :param x: Tensor (N, C, H, W)
    :param low: low-pass filter
    :param high: high-pass filter
    :param dilation:
    :return:
        low: Tensor (N, C, H, W)
        high: Tensor (N, C, H, W)
    """
    if not isinstance(low, torch.Tensor):
        low = torch.tensor(np.copy(np.array(low).ravel()[::-1]),
                           dtype=torch.float, device=low.device)
    if not isinstance(high, torch.Tensor):
        high = torch.tensor(np.copy(np.array(high).ravel()[::-1]),
                            dtype=torch.float, device=high.device)
    shape = [1, 1, 1, 1]
    shape[dim] = low.numel()
    # If filter aren't in the right shape, make them so
    if low.shape != tuple(shape):
        low = low.reshape(*shape)
    if high.shape != tuple(shape):
        high = high.reshape(*shape)

    low_band = F.conv2d(x_pad, low)
    high_band = F.conv2d(x_pad, high)

    return low_band, high_band


def afb2d(x, filts, mode):
    """
    :param x: Tensor (N, C, H, W)
    :param filts:
    :param mode:
    :param dilation:
    :return:
        cA, cH, cV, cD: Tensor (N, C, H, W) four sub bands
    """
    low_col = filts[0].float()
    high_col = filts[1].float()
    low_row = filts[2].float()
    high_row = filts[3].float()
    pad_size = low_col.numel() // 2

    channel = x.shape[1]
    height = x.shape[2]
    width = x.shape[3]
    if channel == 1:
        padding = (pad_size, pad_size, pad_size, pad_size)
    elif channel == 3:
        padding = (pad_size, pad_size, pad_size, pad_size, 0, 0)
    else:
        raise ValueError('channel should be 1 or 3')
    x_pad = wave_pad(x, padding, mode)
    x_pad = wave_pad(x_pad, padding, 'constant')
    low, high = afb1d(x_pad, low_row, high_row, dim=3)
    cA, cH = afb1d(low, low_col, high_col, dim=2)
    cA = cA[:, :, pad_size*2:height+pad_size*2, pad_size*2:width+pad_size*2].clone()
    cH = cH[:, :, pad_size*2:height+pad_size*2, pad_size*2:width+pad_size*2].clone()
    cV, cD = afb1d(high, low_col, high_col, dim=2)
    cV = cV[:, :, pad_size*2:height+pad_size*2, pad_size*2:width+pad_size*2].clone()
    cD = cD[:, :, pad_size*2:height+pad_size*2, pad_size*2:width+pad_size*2].clone()
    return cA, cH, cV, cD


def upconvLOC(band, row_f, col_f, mode='periodic'):
    """
    :param band: Tensor (N, C, H, W)
    :param row_f: row filter
    :param col_f: col filter
    :param mode: default as 'reflect'
    :return:
    """
    height = band.shape[2]
    width = band.shape[3]
    length = row_f.numel()
    pad_size = length // 2
    padding = (pad_size, pad_size, pad_size, pad_size)
    band_pad = wave_pad(band, padding, mode)
    col_recon = F.conv2d(band_pad, col_f)
    recon = F.conv2d(col_recon, row_f)
    recon = recon[:, :, :height, :width]
    return recon


def idwt2LOC(cA, cH, cV, cD, low_row, high_row, low_col, high_col, mode, shift):
    up_a = upconvLOC(cA, low_row, low_col, mode)
    up_h = upconvLOC(cH, low_row, high_col, mode)
    up_v = upconvLOC(cV, high_row, low_col, mode)
    up_d = upconvLOC(cD, high_row, high_col, mode)
    result = up_a + up_h + up_v + up_d
    if shift:
        # col shift
        result1 = torch.zeros(result.shape, device=cA.device)
        temp_row = result[:, :, -1, :].clone()
        result1[:, :, 1:, :] = result[:, :, :-1, :]
        result1[:, :, 0, :] = temp_row

        # row shift
        result2 = torch.zeros(result.shape, device=cA.device)
        temp_col = result1[:, :, :, -1].clone()
        result2[:, :, :, 1:] = result1[:, :, :, :-1]
        result2[:, :, :, 0] = temp_col
        result = result2
    return result


def sfb2d(ll, high_coeffs, filts, level, mode):
    """
    :param ll: cA Tensor (N, C, H, W)
    :param high_coeffs: (cH, cV, cD) Tensor (N, C, H, W)
    :param filts: filters
    :param level: reconstruction level
    :param mode: default 'reflect'
    :return:
        result: Reconstruction result Tensor (N, C, H, W)
    """
    low_col = filts[0].clone().reshape((1, 1, -1, 1)).float()
    high_col = filts[1].clone().reshape((1, 1, -1, 1)).float()
    low_row = filts[2].clone().reshape((1, 1, 1, -1)).float()
    high_row = filts[3].clone().reshape((1, 1, 1, -1)).float()

    step = 2 ** level
    result = ll.clone()
    for i in range(step):
        gap = step * 2
        new_shape = [ll.shape[0], ll.shape[1], ll.shape[2]//step, ll.shape[3]//step]
        cA1 = torch.zeros(new_shape, device=ll.device)
        cA1[:, :, ::2, ::2] = ll[:, :, i::gap, i::gap]
        cH1 = torch.zeros(new_shape, device=ll.device)
        cH1[:, :, ::2, ::2] = high_coeffs[:, 0, i::gap, i::gap].unsqueeze(dim=1)
        cV1 = torch.zeros(new_shape, device=ll.device)
        cV1[:, :, ::2, ::2] = high_coeffs[:, 1, i::gap, i::gap].unsqueeze(dim=1)
        cD1 = torch.zeros(new_shape, device=ll.device)
        cD1[:, :, ::2, ::2] = high_coeffs[:, 2, i::gap, i::gap].unsqueeze(dim=1)
        shift = False
        out1 = idwt2LOC(cA1, cH1, cV1, cD1, low_row, high_row, low_col, high_col, mode, shift)

        cA2 = torch.zeros(new_shape, device=ll.device)
        cA2[:, :, ::2, ::2] = ll[:, :, step+i::gap, step+i::gap]
        cH2 = torch.zeros(new_shape, device=ll.device)
        cH2[:, :, ::2, ::2] = high_coeffs[:, 0, step+i::gap, step+i::gap].unsqueeze(dim=1)
        cV2 = torch.zeros(new_shape, device=ll.device)
        cV2[:, :, ::2, ::2] = high_coeffs[:, 1, step+i::gap, step+i::gap].unsqueeze(dim=1)
        cD2 = torch.zeros(new_shape, device=ll.device)
        cD2[:, :, ::2, ::2] = high_coeffs[:, 2, step+i::gap, step+i::gap].unsqueeze(dim=1)
        shift = True
        out2 = idwt2LOC(cA2, cH2, cV2, cD2, low_row, high_row, low_col, high_col, mode, shift)
        result[:, :, i::step, i::step] = (out1 + out2) * 0.5
    return result



