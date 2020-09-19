import torch
import cv2
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from skimage.measure import compare_psnr, compare_ssim
from networks.FAN import FAN
from utils import load_state_dict_cpu


# option
parser = argparse.ArgumentParser(description='Test on Real-World Noise')
parser.add_argument('--ckpt', type=str, default='./checkpoint/model_real',
                    help="Checkpoint path")
parser.add_argument('--img_dir', type=str, default='./data/SIDD/',
                    help="input directory of images with real-world noise")
parser.add_argument('--is_gt', action='store_true',
                    help="Whether to compare ground truth")
parser.add_argument('--gpu', action='store_true',
                    help="Whether to use GPU")
# model setting
parser.add_argument('--depth_S', type=int, default=5,
                    help="Est-Net depth, default as 5 and do not change")
parser.add_argument('--depth_U', type=int, default=4,
                    help="De-Net layers, default as 4 and do not change")
parser.add_argument('--wave_mode', type=str, default='db1',
                    help="Wave choice for wavelet transform")
args = parser.parse_args()

# load the pre-trained model
print('Loading the model')
checkpoint = torch.load(args.ckpt)
model = FAN(depth_S=args.depth_S, depth_U=args.depth_U, feature_dims=64, wave_pattern=args.wave_mode)
if args.gpu:
    model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(checkpoint)
    print('Begin Testing on GPU')
else:
    load_state_dict_cpu(model, checkpoint)
    print('Begin Testing on CPU')
model.eval()

img_suffixes = (".png", ".bmp", ".jpg", ".jpeg")
img_paths = [str(i) for i in Path(args.img_dir).glob('*noisy*') if i.suffix.lower() in img_suffixes]
for img_path in img_paths:
    img_noisy = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) / 255
    height, width, _ = img_noisy.shape
    if height % 2**(args.depth_U-1) != 0:
        height -= height % 2**(args.depth_U-1)
    if width % 2**(args.depth_U-1) != 0:
        width -= width % 2**(args.depth_U-1)
    img_noisy = img_noisy[:height, :width, :]
    img_noisy_input = torch.from_numpy(img_noisy.transpose(2, 0, 1)).unsqueeze(dim=0)

    if args.gpu:
        img_noisy_input = img_noisy_input.cuda()
    with torch.autograd.set_grad_enabled(False):
        tic = time.time()
        pred_map, img_denoise = model(img_noisy_input.float())
        toc = time.time()
    print('Finish! The spending time is {:.2f}'.format(toc- tic))
    if args.gpu:
        img_denoise = img_denoise.cpu().numpy()
    else:
        img_denoise = img_denoise.numpy()
    img_denoise = np.clip(img_denoise.squeeze(), 0.0, 1.0)
    img_denoise = img_denoise.transpose(1, 2, 0)

    # ground truth
    if args.is_gt:
        gt_path = img_path.replace('noisy', 'gt')
        if not Path(gt_path).exists():
            raise ValueError('No such file, please set is_gt False')
        img_gt = cv2.cvtColor(cv2.imread(gt_path), cv2.COLOR_BGR2RGB) / 255

        psnr = compare_psnr(img_gt, img_denoise, data_range=1)
        ssim = compare_ssim(img_gt, img_denoise, data_range=1, gaussian_weights=True,
                            use_sample_covariance=False, multichannel=True)
        print('PSNR is {:.2f}'.format(psnr))
        print('SSIM is {:.2f}'.format(ssim))

        plt.subplot(131)
        plt.imshow(img_noisy)
        plt.title('Noisy Image')
        plt.subplot(132)
        plt.imshow(img_gt)
        plt.title('Ground Truth')
        plt.subplot(133)
        plt.imshow(img_denoise)
        plt.title('Denoised Image')
        plt.show()
    else:
        plt.subplot(121)
        plt.imshow(img_noisy)
        plt.title('Noisy Image')
        plt.subplot(122)
        plt.imshow(img_denoise)
        plt.title('Denoised Image')
        plt.show()
