import numpy as np
import torch
import torch.utils.data
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def PSNR(input, target):
    output = []
    with torch.no_grad():
        for i in range(input.size(0)):
            input_ = input[i]
            target_ = target[i]
            input_ = input_.cpu().detach().numpy()
            target_ = target_.cpu().detach().numpy()
            input_ = ((np.clip(input_, -1, 1) + 1) * 127.5).astype(np.uint8)
            target_ = ((np.clip(target_, -1, 1) + 1) * 127.5).astype(np.uint8)
            output.append(psnr(input_, target_))
    return np.mean(output)

def SSIM(input, target):
    output = []
    with torch.no_grad():
        for i in range(input.size(0)):
            input_ = input[i]
            target_ = target[i]
            input_ = input_.cpu().detach().numpy()
            target_ = target_.cpu().detach().numpy()
            input_ = ((np.clip(input_, 0, 1) + 1) * 127.5).astype(np.uint8)
            target_ = ((np.clip(target_, 0, 1) + 1) * 127.5).astype(np.uint8)
            output.append(ssim(input_, target_, channel_axis=0))
    return np.mean(output)

def NRMSE(input, target):
    output = []
    with torch.no_grad():
        for i in range(input.size(0)):
            input_ = input[i]
            target_ = target[i]
            input_ = input_.cpu().detach().numpy()
            target_ = target_.cpu().detach().numpy()
            input_ = ((np.clip(input_, 0, 1) + 1) * 127.5).astype(np.uint8)
            target_ = ((np.clip(target_, 0, 1) + 1) * 127.5).astype(np.uint8)
            output.append(np.sqrt(np.mean((input_ - target_) ** 2)) / (np.max(target_)-np.min(target_)))
    return np.mean(output)