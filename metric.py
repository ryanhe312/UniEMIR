import os
import sys
import tqdm
import numpy as np
import tifffile as tiff
from models.csbdeep.utils import normalize
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

path = sys.argv[1] + "/results/test/0"

GT_paths = sorted([f for f in os.listdir(path) if 'GT' in f])
Output_paths = sorted([f for f in os.listdir(path) if 'Out' in f])

psnrs, ssims, nrmse = [], [], []
for out, gt in tqdm.tqdm(zip(Output_paths, GT_paths)):
    out = os.path.join(path, out)
    gt = os.path.join(path, gt)

    out = tiff.imread(out)
    gt = tiff.imread(gt)

    psnrs.append(psnr(out, gt))
    ssims.append(ssim(out, gt))
    nrmse.append(np.sqrt(np.mean((out - gt) ** 2)) / (np.max(gt)-np.min(gt)))

# save psnr and ssim as txt
np.savetxt(os.path.join(path, 'psnr.txt'), psnrs)
np.savetxt(os.path.join(path, 'ssim.txt'), ssims)
np.savetxt(os.path.join(path, 'nrmse.txt'), nrmse)

print(f'PSNR mean: {sum(psnrs) / len(psnrs)}, variance: {np.var(psnrs)}')
print(f'SSIM mean: {sum(ssims) / len(ssims)}, variance: {np.var(ssims)}')
print(f'NRMSE mean: {sum(nrmse) / len(nrmse)}, variance: {np.var(nrmse)}')