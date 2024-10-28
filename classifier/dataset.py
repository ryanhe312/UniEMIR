import os
import random
import numpy as np
import torch
import torch.multiprocessing
import torch.utils.data as data
from PIL import Image
from PIL import ImageOps
from torchvision import transforms
import cv2
from noise import add_gaussian_noise


def find_max_number(folder_path):
    max_number = 0
    for filename in os.listdir(folder_path):
        if filename.endswith('.tif'):
            filename = filename[:-4]
        if not filename.isdigit():
            continue
        filename = int(filename)

        number = int(filename)
        max_number = max(max_number, number)
    return max_number


def find_max_folder_number(folder_path):
    max_number = 0
    for folder_name in os.listdir(folder_path):
        if os.path.isdir(os.path.join(folder_path, folder_name)):
            if not folder_name.isdigit():
                continue
            folder_name = int(folder_name)
            number = int(folder_name)
            max_number = max(max_number, number)
    return max_number


def pil_loader(path):
    return Image.open(path).convert('L')


def pil_loader_noL(path):
    return Image.open(path)


def invert(tensor):
    return 1 - tensor


class EMDiffusenDataset(data.Dataset):  # Denoise and super-resolution Dataset
    def __init__(self, data_root, data_len=-1, norm=True, percent=False, phase='train', image_size=[256, 256], pssr=False,
                 loader=pil_loader, corrupt='noise'):
        self.data_root = data_root
        self.phase = phase
        if pssr:
            self.img_paths, self.gt_paths = self.read_dataset_pssr(self.data_root)
        else:
            self.img_paths, self.gt_paths = self.read_dataset(self.data_root)
        if percent:
            self.img_paths = self.img_paths[:int(len(self.img_paths) * percent)]
            self.gt_paths = self.gt_paths[:int(len(self.gt_paths) * percent)]
        self.tfs = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.loader = loader
        self.norm = norm
        self.image_size = image_size
        self.corrupt = corrupt
        self.one_hot = [torch.tensor([0, 0, 0, 0, 0], dtype=torch.float32), torch.tensor([1, 0, 0, 0, 0], dtype=torch.float32),
                        torch.tensor([0, 1, 0, 0, 0], dtype=torch.float32), torch.tensor([0, 0, 1, 0, 0], dtype=torch.float32),
                        torch.tensor([0, 0, 0, 1, 0], dtype=torch.float32), torch.tensor([0, 0, 0, 0, 1], dtype=torch.float32),
                        ]

    def __getitem__(self, index):
        ret = {}
        gt_file_name = self.gt_paths[index]
        img = self.loader(gt_file_name) # 仅使用 gt 进行降质
        if self.phase == 'train':
            img = self.aug(img)
        img = np.array(img)

        if self.corrupt == 'blur':
            kernel_sizes = [0, 3, 5, 7, 9, 11]
            index = random.randint(0, len(kernel_sizes) - 1)
            kernel_size = kernel_sizes[index]
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0) if kernel_size != 0 else img
            label = self.one_hot[index]
        elif self.corrupt == 'noise':
            noise_levels = [0, 10, 20, 30, 40, 50]
            index = random.randint(0, len(noise_levels) - 1)
            noise_level = noise_levels[index]
            img = add_gaussian_noise(img, noise_level)
            label = self.one_hot[index]

        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        img = 20 * np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]) + 1)

        img = (((img - np.min(img)) / (np.max(img) - np.min(img))) * 255).astype(np.uint8)
        img = Image.fromarray(img)
        img = self.tfs(img)

        return img.repeat(3, 1, 1), label, gt_file_name

    def __len__(self):
        return len(self.img_paths)

    def aug(self, img):
        if random.random() < 0.5:
            img = ImageOps.flip(img)
        if random.random() < 0.5:
            img = img.rotate(90)
        return img

    def read_dataset(self, data_root):
        import os
        img_paths = []
        gt_paths = []
        for cell_num in os.listdir(data_root):
            if cell_num == '.DS_Store':
                continue
            cell_path = os.path.join(data_root, cell_num)
            for noise_level in os.listdir(cell_path):
                for img in sorted(os.listdir(os.path.join(cell_path, noise_level))):
                    if 'tif' in img:
                        img_paths.append(os.path.join(cell_path, noise_level, img))
                        gt_paths.append(os.path.join(cell_path, noise_level, img).replace('wf', 'gt'))
        return img_paths, gt_paths
    
    def read_dataset_pssr(self, data_root):
        import os
        img_paths = []
        gt_paths = []
        for img in sorted(os.listdir(data_root)):
            if 'tif' in img:
                gt_paths.append(os.path.join(data_root, img))
                img_paths.append(os.path.join(data_root, img).replace('HR', 'LR').replace('_hr', '_lr'))
        return img_paths, gt_paths


