import torch.nn as nn
import pywt
from torch_wavelet.utils import prep_filt, afb2d, sfb2d, upsample


class SWTForward(nn.Module):
    def __init__(self, wave='db1', level=1, mode='periodic'):
        super(SWTForward, self).__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            low_col, high_col = wave.dec_lo, wave.dec_hi
            low_row, high_row = low_col, high_col
        else:
            if len(wave) == 2:
                low_col, high_col = wave[0], wave[1]
                low_row, high_row = low_col, high_col
            elif len(wave) == 4:
                low_col, high_col = wave[0], wave[1]
                low_row, high_row = wave[2], wave[3]

        # Prepare the filters
        low_col, high_col, low_row, high_row = \
            low_col[::-1], high_col[::-1], low_row[::-1], high_row[::-1]
        low_col, high_col, low_row, high_row = \
            prep_filt(low_col, high_col, low_row, high_row)
        # add filters to the network for using F.conv2d (input and weight should be the same dtype)
        self.low_col = nn.Parameter(low_col, requires_grad=False)
        self.high_col = nn.Parameter(high_col, requires_grad=False)
        self.low_row = nn.Parameter(low_row, requires_grad=False)
        self.high_row = nn.Parameter(high_row, requires_grad=False)
        self.mode = mode
        self.level = level

    def forward(self, x):
        """
        :param x: Tensor (N, C, H, W)
        :return: ll: Tensor (N, C, H, W)
                 high_coeffs: List with length of 3*level, each element is Tensor (N, C, H, W)
        """
        filts = [self.low_col, self.high_col, self.low_row, self.high_row]
        ll = x
        high_coeffs = []
        for j in range(self.level):
            y = afb2d(ll, filts, self.mode)
            high_coeffs += y[1:]
            ll = y[0]
            filts = upsample(filts)

        return ll, high_coeffs


class SWTInverse(nn.Module):
    def __init__(self, wave='db1', level=1, mode='periodic'):
        super(SWTInverse, self).__init__()
        if isinstance(wave, str):
            wave = pywt.Wavelet(wave)
        if isinstance(wave, pywt.Wavelet):
            low_col, high_col = wave.dec_lo, wave.dec_hi
            low_row, high_row = low_col, high_col
        else:
            if len(wave) == 2:
                low_col, high_col = wave[0], wave[1]
                low_row, high_row = low_col, high_col
            elif len(wave) == 4:
                low_col, high_col = wave[0], wave[1]
                low_row, high_row = wave[2], wave[3]

        # Prepare the filters
        low_col, high_col, low_row, high_row = \
            prep_filt(low_col, high_col, low_row, high_row)
        self.low_col = nn.Parameter(low_col, requires_grad=False)
        self.high_col = nn.Parameter(high_col, requires_grad=False)
        self.low_row = nn.Parameter(low_row, requires_grad=False)
        self.high_row = nn.Parameter(high_row, requires_grad=False)
        self.level = level
        self.mode = mode

    def forward(self, x):
        """
        :param x: Coeff (ll, high_coeffs)
                  each sub band shape (N, C, H, W)
                  ll: Tensor (N, C, H, W)
                  high_coeffs: Tensor (N, level*3, H, W)
        :return:
                Tensor (N, C, H, W)
        """
        filts = (self.low_col, self.high_col, self.low_row, self.high_row)
        ll = x[0]
        lohi = x[1]
        for i in range(self.level-1, -1, -1):
            lohi_level = lohi[:, i*3:(i+1)*3]
            ll = sfb2d(ll, lohi_level, filts, i, self.mode)
        return ll
