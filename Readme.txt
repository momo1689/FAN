This code is for Frequency Attention Networks.

Requirements and Dependencies:
Please install requirements.txt by pip

Test Additive White Gaussian Noise(AWGN):
python test_awgn.py --ckpt[trained model] --img_dir[dataset directory] --sigma[sigma] --gpu[GPU or CPU]

Test Real-World Noise:
There are two datasets in this folder, including SIDD and DND. We save them as .png in ./data/SIDD and ./data/DND while the SIDD dataset has ground truth, DND not.
python test_real_noise.py --ckpt[trained model] --img_dir[dataset directory] --is_gt[whether to compare ground truth(please set False when test the DND dataset)] --gpu[GPU or CPU]

If you want to test AWGN, please run the following scripts:
python test_awgn.py --img_dir ./data/LIVE1/ --gpu

If you want to test SIDD dataset, please run the following scripts:
python test_real_noise.py --img_dir ./data/SIDD/ --gpu --is_gt

If you want to test DND dataset, please run the following scripts:
python test_real_noise.py --img_dir ./data/DND/ --gpu
