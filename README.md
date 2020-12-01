# Frequency Attention Network
This repository is for Frequency Attention Network(FAN) introduced in the following paper

![Frequency Attention Network: Blind Noise Removal for Real Images, ACCV, 2020](https://openaccess.thecvf.com/content/ACCV2020/papers/Mo_Frequency_Attention_Network_Blind_Noise_Removal_for_Real_Images_ACCV_2020_paper.pdf)

## Requirement and Dependecies

* Python 3.6, torch 1.1.0
* More details(see requirements.txt)

## Abstract

With outstanding feature extraction capabilities, deep convolutional neural networks(CNNs) have achieved extraordinary improvements in image denoising tasks. However, because of the difference of statistical characteristics of signal-dependent noise and signal-independent noise, it is hard to model real noise for training and blind real image denoising is still an important challenge problem. In this work we propose a method for blind image denoising that combines frequency domain analysis and attention mechanism, named frequency attention network(FAN). We adopt wavelet transform to convert images from spatial domain to frequency domain with more sparse features to utilize spectral information and structure information. For the denoising task, the objective of the neural network is to estimate the optimal solution of the wavelet coeffcients of the clean image by nonlinear characteristics, which makes FAN possess good interpretability. Meanwhile, spatial and channel mechanisms are employed to enhance feature maps at different scales for capturing contextual information. Extensive experiments on the synthetic noise dataset and two real-world noise benchmarks indicate the superiority of our method over other competing methods at different noise type cases in blind image denoising.

## Network Architecture

![FAN Architecture](https://github.com/momo1689/FAN/blob/master/figs/network.png)

Our proposed method FAN contains two subnetwork -- Est-Net and De-Net. Est-Net is to estimate the noise level map which plays the role of guidance. De-Net is for image denoising.

![Spatial-Channel Attention Block](https://github.com/momo1689/FAN/blob/master/figs/SCAB.png)

The SCAB combines the spatial attention mechanism and channel attention mechanism. The spatial mechanism can reweight the feature maps based on the position of different feature maps while the channel mechanism can focus on different types of features.

## Results on AWGN

![Test on sigma=50](https://github.com/momo1689/FAN/blob/master/figs/awgn.png)

Test on the LIVE1 dataset with the addiative gaussian white noise of sigma = 50.

## Results on Real Noise Dataset

![Test on SIDD Dataset](https://github.com/momo1689/FAN/blob/master/figs/SIDD.png)

One example of test on SIDD dataset.

![Test on DND Dataset](https://github.com/momo1689/FAN/blob/master/figs/DND.png)

One example of test on DND dataset.

## Code User Guide

### Test Additive White Gaussian Noise(AWGN)

python test_awgn.py --ckpt[trained model] --img_dir[dataset directory] --sigma[sigma] --gpu[GPU or CPU]

### Test Real-World Noise

python test_real_noise.py --ckpt[trained model] --img_dir[dataset directory] --is_gt[whether to compare ground truth(please set False when test the DND dataset)] --gpu[GPU or CPU]
