import os
import sys
import tqdm
import tifffile as tiff
import numpy as np

path = sys.argv[1]

name_list = [ 'Input_lower', 'Out_0', 'Out_1', 'Out_2', 'Out_3', 'Out_4'] # test Isotropic
name_list = [ 'Out_0'] # test SR/Denoising

path_lists = [sorted(
        [f for f in os.listdir(path) if name in f], 
        key=lambda x:(int(x.split('_')[4]),int(x.split('_')[5]),int(x.split('.')[0].split('_')[6]))
    ) for name in name_list
]

volume = {}
max_x, max_y = 0, 0
for i in range(len(path_lists[0])):
    _, _, _, _, layer_idx, x_idx, y_idx = path_lists[0][i].split('.')[0].split('_')
    img = []
    for j in range(len(path_lists)):
        img.append(tiff.imread(os.path.join(path, path_lists[j][i])))
        print(os.path.join(path, path_lists[j][i]))
    if (int(x_idx), int(y_idx)) not in volume:
        volume[(int(x_idx), int(y_idx))] = []
        max_x = max(max_x, int(x_idx))
        max_y = max(max_y, int(y_idx))
    volume[(int(x_idx), int(y_idx))] += img

patch_size = volume[(0, 0)][0].shape
stride = 224 # test Isotropic
# stride = 256 # test Segmentation and Anisotropic
volume_merged = np.zeros((len(volume[(0, 0)]), stride * max_x + patch_size[0], stride * max_y + patch_size[1]))
for key in volume.keys():
    volume_merged[:, 
        key[1] * stride: key[1] * stride + patch_size[1],
        key[0] * stride: key[0] * stride + patch_size[0], 
    ] = np.stack(volume[key], axis=0)

tiff.imwrite(os.path.join(path, 'volume.tif'), volume_merged.astype(np.uint8))