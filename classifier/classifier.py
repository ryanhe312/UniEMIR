import torch
import numpy as np
import cv2
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image

def add_gaussian_noise(image, noise_level):
    mean = 0
    std_dev = noise_level
    gauss = np.random.normal(mean, std_dev, image.shape)
    noisy = image.astype(np.float32) + gauss
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8)

def add_gaussian_blur(img, kernel_size):
    img = np.array(img)
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def fft(img):
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    return 20 * np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]) + 1)

def ndarry2tensor(img):
    tfs = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    img = (((img - np.min(img)) / (np.max(img) - np.min(img))) * 255).astype(np.uint8)
    img = Image.fromarray(img)
    if len(img.size) < 4:
        img = tfs(img)
        if img.size(0) == 1:
            img = img.repeat(3, 1, 1)
        img = img.unsqueeze(0)
    return img

def fft_tensor(img):
    # center crop 256
    h, w = img.shape
    img = img[h//2-128:h//2+128, w//2-128:w//2+128]
    return ndarry2tensor(fft(img))

class Classifier: 

    def __init__(self, task='noise', device=None):
        self.model = models.resnet18()
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 5)
        if task == 'noise':
            checkpoint = torch.load(r"classifier/noise_best.pth", weights_only=True)
        else:
            checkpoint = torch.load(r"classifier/blur_best.pth", weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if device == None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.model.to(device)
        self.model.eval()
        self.task = task
        if self.task == 'noise':
            self.label_list = [10, 20, 30, 40, 50]
        else:
            self.label_list = [3, 5, 7, 9, 11]
        
    def __call__(self, input):
        ''' input: spectrum \n '''
        with torch.no_grad():
            input = input.to(self.device)
            outputs = self.model(input)
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.to('cpu')
            
            results = self.label_list[predicted]
        return results, outputs
        
if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    c_noise = Classifier(task='noise', device=device)
    c_blur = Classifier(task='blur', device=device)

    img = Image.open(r'/home/user2/dataset/microscope/EMDiffuse/EMDiffuse_dataset/Liver_train/denoise/train_gt/1/Liver__4w_05/1.tif').convert('L')
    noise_levels = [10, 20, 30, 40, 50]
    noise_level = np.random.choice(noise_levels, 1)[0]
    print(noise_level)
    img = np.array(img)
    img_noise = add_gaussian_noise(img, noise_level)
    results, outputs = c_noise(ndarry2tensor(fft(img_noise)))
    outputs = outputs.to("cpu")
    print("noise:", outputs, "noise level:", results)

    img = Image.open(r'/home/user2/dataset/microscope/EMDiffuse/EMDiffuse_dataset/brain_train/zoom/train_gt/10/Brain__2w_01/1.tif').convert('L')
    kernel_sizes = [3, 5, 7, 9, 11]
    kernel_size = np.random.choice(kernel_sizes, 1)[0]
    img = np.array(img)
    print(kernel_size)
    img_blur = add_gaussian_blur(img, kernel_size)
    results, outputs = c_blur(ndarry2tensor(fft(img_blur)))
    outputs = outputs.to("cpu")
    print("blur:", outputs, "kernel size:", results)

