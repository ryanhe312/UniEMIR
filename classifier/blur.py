import cv2
import numpy as np

def normalize_spectrum(spectrum):
    spectrum_normalized = ((spectrum - np.min(spectrum)) / (np.max(spectrum) - np.min(spectrum))) * 255
    return spectrum_normalized.astype(np.uint8)

def save_spectrum(spectrum, save_path):
    spectrum_normalized = normalize_spectrum(spectrum)
    cv2.imwrite(save_path, spectrum_normalized)

def main():
    img_path = r'/home/user2/dataset/microscope/EMDiffuse/EMDiffuse_dataset/brain_train/zoom/train_gt/4/Brain__2w_01/49.tif' 
    original_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) 

    kernel_sizes = [0, 3, 5, 7, 9, 11]
    blurred_images = [cv2.GaussianBlur(original_image, (i, i), 0) if i != 0 else original_image for i in kernel_sizes]

    ffts = []
    for blurred_image in blurred_images:
        dft = cv2.dft(np.float32(blurred_image), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]) + 1)
        ffts.append(magnitude_spectrum)
    
    pic_raws = np.hstack(blurred_images)
    pic_ffts = np.hstack(ffts)
    pic = np.vstack((pic_raws, pic_ffts))
    save_spectrum(pic, "blur.png")

if __name__ == "__main__":
    main()