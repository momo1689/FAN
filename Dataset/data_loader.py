import torch
import h5py
import cv2
import numpy as np
from torch.utils import data
from Dataset.dataAug import data_augment
from skimage.util import random_noise


def noise_estimate(noisy_img, gt_img, win=9, spatial_sigma=7):
    diff = abs(noisy_img.astype(np.float) - gt_img.astype(np.float))
    blur = cv2.GaussianBlur(diff, (win, win), spatial_sigma)
    sigma_map_est = np.clip(blur, 1e-10, 1.0)
    return sigma_map_est


# for real-world noise training
class DataFromH5File(data.Dataset):
    def __init__(self, filepath):
        h5file = h5py.File(filepath, 'r')
        self.gt = h5file['gt']
        self.noisy = h5file['noisy']

    def __getitem__(self, idx):
        gt_img = self.gt[idx] / 255
        noisy_img = self.noisy[idx] / 255
        # data augment
        mode = np.random.randint(0, 6)
        noisy_img, gt_img = data_augment(noisy_img, gt_img, mode)
        sigma_map = noise_estimate(noisy_img, gt_img)

        gt = torch.from_numpy(gt_img.transpose(2, 0, 1).copy()).float()
        noisy = torch.from_numpy(noisy_img.transpose(2, 0, 1).copy()).float()
        smap = torch.from_numpy(sigma_map.transpose(2, 0, 1).copy()).float()
        return noisy, gt, smap

    def __len__(self):
        return self.gt.shape[0]


# for synthetic noise training
class DataFromH5AWGN(data.Dataset):
    def __init__(self, filepath):
        h5file = h5py.File(filepath, 'r')
        self.gt = h5file['gt']

    def __getitem__(self, idx):
        gt_img = self.gt[idx] / 255
        # noisy image generate
        sigma = np.random.randint(15, 51)
        noisy_img = random_noise(gt_img, 'gaussian', mean=0, var=(sigma/255)**2, clip=True)
        # data augment
        mode = np.random.randint(0, 6)
        noisy_img, gt_img = data_augment(noisy_img, gt_img, mode)
        sigma_map = noise_estimate(noisy_img, gt_img)

        gt = torch.from_numpy(gt_img.transpose(2, 0, 1).copy()).float()
        noisy = torch.from_numpy(noisy_img.transpose(2, 0, 1).copy()).float()
        smap = torch.from_numpy(sigma_map.transpose(2, 0, 1).copy()).float()
        return noisy, gt, smap

    def __len__(self):
        return self.gt.shape[0]
