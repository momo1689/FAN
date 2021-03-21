import numpy as np


def data_augment(noisy, gt, mode):
    if mode == 0:
        return noisy, gt
    elif mode == 1:
        noisy = np.flipud(noisy)
        gt = np.flipud(gt)
        return noisy, gt
    elif mode == 2:
        noisy = np.fliplr(noisy)
        gt = np.fliplr(gt)
        return noisy, gt
    elif mode == 3:
        noisy = np.rot90(noisy)
        gt = np.rot90(gt)
        return noisy, gt
    elif mode == 4:
        noisy = np.rot90(np.rot90(noisy))
        gt = np.rot90(np.rot90(gt))
        return noisy, gt
    elif mode == 5:
        noisy = np.rot90(np.rot90(np.rot90(noisy)))
        gt = np.rot90(np.rot90(np.rot90(gt)))
        return noisy, gt
