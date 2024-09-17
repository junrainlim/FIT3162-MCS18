import cv2
import numpy as np


# Function to calculate MSE (Mean Squared Error)
def calculate_mse(image1, image2):
    # Ensure the images have the same dimensions
    assert image1.shape == image2.shape, "Images must have the same dimensions."
    
    # Convert to float and calculate squared differences
    squared_diff = (image1.astype(np.float64) - image2.astype(np.float64)) ** 2
    
    # Compute MSE
    mse = np.mean(squared_diff)
    return mse

# Function to calculate PSNR
def calculate_psnr(image1, image2):
    mse = calculate_mse(image1, image2)
    if mse == 0:
        return float('inf')  # No difference, PSNR is infinite
    
    # MAX pixel value for 8-bit image is 255
    max_pixel_value = 255.0
    psnr = 10 * np.log10((max_pixel_value ** 2) / mse)
    return psnr





# Load the original image
original_img = cv2.imread("images/nando-jpeg-quality-001.jpg")
print(f"Original Image: {type(original_img)}, Shape: {original_img.shape if original_img is not None else 'None'}")

# Load the decrypted image
decrypted_img = cv2.imread("images/decrypted_image.jpg")
print(f"Decrypted Image: {type(decrypted_img)}, Shape: {decrypted_img.shape if decrypted_img is not None else 'None'}")

# PSNR testing between original and encrypted image
psnr_value_decrypted = calculate_psnr(original_img, decrypted_img)
print(f"PSNR between original and decrypted image: {psnr_value_decrypted} dB")



