import cv2 as cv
import numpy as np
import hashlib
import time

start_time = time.time()

# Read the scrambled image
img_path = "images/image_scrambled.png"
img_scrambled = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
assert img_scrambled is not None, "Image not found or unable to load."

# Define block size
block_size = 8

# Step 1 : Divide image into blocks of 8x8 pixels
blocks = [
    img_scrambled[i: i + block_size, j: j + block_size]
    for i in range(0, img_scrambled.shape[0], block_size)
    for j in range(0, img_scrambled.shape[1], block_size)
]


# Generate the same keys used in encryption
K1 = hashlib.sha256(b"secret_key_1").digest()
K2 = hashlib.sha256(b"secret_key_2").digest()
K3 = hashlib.sha256(b"secret_key_3").digest()

# Use keys to seed the random number generators
rng_1 = np.random.default_rng(int.from_bytes(K1, byteorder='big'))
rng_2 = np.random.default_rng(int.from_bytes(K2, byteorder='big'))
rng_3 = np.random.default_rng(int.from_bytes(K3, byteorder='big'))


# negative_positive_transformation function
def negative_positive_transformation(block, L=8):
    return 255 - block  # assuming 8-bit grayscale image



# Define transformation functions
def undo_rotate_block(block, angle):
    if angle == 0:
        return block
    elif angle == 90:
        return cv.rotate(block, cv.ROTATE_90_COUNTERCLOCKWISE)
    elif angle == 180:
        return cv.rotate(block, cv.ROTATE_180)
    elif angle == 270:
        return cv.rotate(block, cv.ROTATE_90_CLOCKWISE)


# invert block function
def invert_block(block):
    return cv.flip(block, 1)


# Step 1.5: Apply negative-positive transformation using K3
np_blocks = []
for block in blocks:
    if rng_3.random() > 0.5:
        block = negative_positive_transformation(block)
    np_blocks.append(block)

print(f"Shape of the first block in np_blocks: {np_blocks[0].shape}")

# Step 2: Rotate and invert block using key K2
transformed_blocks = []
for block in np_blocks:
    angle = rng_2.choice([0, 90, 180, 270])  # Random rotation
    if rng_2.random() > 0.5:  # 0.5 value suggested from paper
        block = invert_block(block)  # Random inversion
    block = undo_rotate_block(block, angle)
    transformed_blocks.append(block)

print(f"Shape of the first block in transformed_blocks: {transformed_blocks[0].shape}")


# Step 3: Assemble blocks with key K1 (inverse permutation)
transformed_blocks = rng_1.permutation(transformed_blocks)



# Reconstruct blocks into image
width = img_scrambled.shape[1] // block_size
rows = [cv.hconcat(transformed_blocks[i: i + width]) for i in range(0, len(transformed_blocks), width)]
img_reconstructed = cv.vconcat(rows)


# Step 4: Separate the grayscale-based image into three channels Y, Cb, and Cr
channel_width = img_reconstructed.shape[1] // 3
Y_channel = img_reconstructed[:, :channel_width]
Cb_channel = img_reconstructed[:, channel_width:2 * channel_width]
Cr_channel = img_reconstructed[:, 2 * channel_width:]

# Verify the shape and alignment of each channel
print(f"Y_channel shape: {Y_channel.shape}")
print(f"Cb_channel shape: {Cb_channel.shape}")
print(f"Cr_channel shape: {Cr_channel.shape}")

# if channels are in 3 combine them into 1
if len(Y_channel.shape) == 3 and Y_channel.shape[2] == 3:
    Y_channel = cv.cvtColor(Y_channel, cv.COLOR_BGR2GRAY)
if len(Cb_channel.shape) == 3 and Cb_channel.shape[2] == 3:
    Cb_channel = cv.cvtColor(Cb_channel, cv.COLOR_BGR2GRAY)
if len(Cr_channel.shape) == 3 and Cr_channel.shape[2] == 3:
    Cr_channel = cv.cvtColor(Cr_channel, cv.COLOR_BGR2GRAY)

# Step 5: Transform the three channels to RGB color channels
img_YCC_restored = cv.merge([Y_channel, Cb_channel, Cr_channel])


# Check the number of channels in each Y, Cb, and Cr channel
print(f"Y_channel shape: {Y_channel.shape}, Channels: {Y_channel.shape[2] if len(Y_channel.shape) > 2 else 1}")
print(f"Cb_channel shape: {Cb_channel.shape}, Channels: {Cb_channel.shape[2] if len(Cb_channel.shape) > 2 else 1}")
print(f"Cr_channel shape: {Cr_channel.shape}, Channels: {Cr_channel.shape[2] if len(Cr_channel.shape) > 2 else 1}")



# Step 6: Combine the RGB channels to generate the decrypted image
img_restored = cv.cvtColor(img_YCC_restored, cv.COLOR_YCrCb2BGR)

# Save the reconstructed image
cv.imwrite("output/img_restored.png", img_restored)

# Display the reconstructed image
cv.imshow("Display Window", img_restored)
cv.waitKey(0)
cv.destroyAllWindows()

end_time = time.time()

print(f"Time taken for decryption: {end_time - start_time} seconds")
