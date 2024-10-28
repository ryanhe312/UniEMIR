import cv2
import numpy as np

def normalize_spectrum(spectrum):
    spectrum_normalized = ((spectrum - np.min(spectrum)) / (np.max(spectrum) - np.min(spectrum))) * 255
    return spectrum_normalized.astype(np.uint8)

def save_spectrum(spectrum, save_path):
    spectrum_normalized = normalize_spectrum(spectrum)
    cv2.imwrite(save_path, spectrum_normalized)

def add_gaussian_noise(image, noise_level):
    row, col = image.shape
    mean = 0
    std_dev = noise_level
    
    gauss = np.random.normal(mean, std_dev, (row, col))
    noisy = image + gauss
    
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8)

def main():
    img_path = r'/home/user2/dataset/microscope/EMDiffuse/EMDiffuse_dataset/brain_train/zoom/train_gt/4/Brain__2w_01/49.tif' 
    original_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) 

    noise_levels = [0, 10, 20, 30, 40, 50, 100]
    noised_images = [add_gaussian_noise(original_image, i) if i != 0 else original_image for i in noise_levels]
    
    ffts = []
    for noised_image in noised_images:
        dft = cv2.dft(np.float32(noised_image), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]) + 1)
        ffts.append(magnitude_spectrum)
    
    pic_raws = np.hstack(noised_images)
    pic_ffts = np.hstack(ffts)
    pic_noise = np.hstack([img - original_image for img in noised_images])
    pic = np.vstack((pic_raws, pic_ffts, pic_noise))
    save_spectrum(pic, "noise.png")

if __name__ == "__main__":
    main()