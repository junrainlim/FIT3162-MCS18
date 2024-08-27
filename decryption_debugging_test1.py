import cv2 as cv
import numpy as np
import hashlib
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend

# Read the scrambled image
img_path = "images/image_scrambled.png"
img_scrambled = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
assert img_scrambled is not None, "Image not found or unable to load."



block_size = 8
num_blocks = (img_scrambled.shape[0] // block_size) * (img_scrambled.shape[1] // block_size)

# Step 1 : Divide image into blocks of 8x8 pixels

blocks = [
    img_scrambled[i: i + block_size, j: j + block_size]
    for i in range(0, img_scrambled.shape[0], block_size)
    for j in range(0, img_scrambled.shape[1], block_size)
]



master_key = "gcdfe".encode()
# Salt (can be any random value)
salt = "abcd".encode()

def derive_key(bytevalue):
    hkdf = HKDF(algorithm=hashes.SHA256(), length=32, salt=salt, info=bytevalue, backend=default_backend())
    return hkdf.derive(master_key)



# Generate a complex key
K1 = derive_key("12345".encode())
K2 = derive_key("23456".encode())
K3 = derive_key("13579".encode())



# Function to generate a deterministic sequence for negative positive transformation
def generate_negative_positive_flags(key: int, num_blocks: int) -> np.ndarray:
    key_int = int.from_bytes(key, byteorder = 'big')
    negative_positive_sequence = np.zeros(num_blocks, dtype=bool)
    for i in range(num_blocks):
        negative_positive_sequence[i] = (key_int + i) % 2 == 0
    return negative_positive_sequence

# Reverse negative positive transformation 
def reverse_negative_positive_transformation(block, L=8):
    return 255 - block  # reverse is the same operation

# Step 1: Reverse the Negative-Positive Transformation using key K3
negative_positive_sequence = generate_negative_positive_flags(K3, num_blocks)
transformed_blocks = []
for i, block in enumerate(blocks):
    if negative_positive_sequence[i]:
         block = reverse_negative_positive_transformation(block)
    transformed_blocks.append(block)



# Define transformation functions
def rotate_block(block, angle):
    if angle == 0:
        return block
    elif angle == 90:
        return cv.rotate(block, cv.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv.rotate(block, cv.ROTATE_180)
    elif angle == 270:
        return cv.rotate(block, cv.ROTATE_90_COUNTERCLOCKWISE)
    

# invert block function
def invert_block(block):
    return cv.flip(block, 1)


# Function to generate a deterministic sequence for rotation angles
def generate_rotation_angles(key: int, num_blocks: int) -> np.ndarray:
    key_int = int.from_bytes(key, byteorder='big')
    angles = np.array([0, 90, 180, 270])
    rotation_sequence = np.zeros(num_blocks, dtype=int)
    for i in range(num_blocks):
        rotation_sequence[i] = angles[(key_int + i) % 4]
    return rotation_sequence


# Function to generate a deterministic sequence for inversion decisions
def generate_inversion_flags(key: int, num_blocks: int) -> np.ndarray:
    key_int = int.from_bytes(key, byteorder='big')
    inversion_sequence = np.zeros(num_blocks, dtype=bool)
    for i in range(num_blocks):
        inversion_sequence[i] = ((key_int + i) % 2) == 0  # Alternating inversion
    return inversion_sequence


rotation_sequence = generate_rotation_angles(K2, num_blocks)
inversion_sequence = generate_inversion_flags(K2, num_blocks)

reversed_blocks = []
for i, block in enumerate(transformed_blocks):
    # Reverse the rotation 
    angle = (360 - rotation_sequence[i]) % 360
    block = rotate_block(block, angle)

    # Reverse the inversion
    if inversion_sequence[i]:
        block = invert_block(block)

    reversed_blocks.append(block)

# Generate deterministic permutation for block shuffling
def generate_deterministic_permutation(key, num_blocks):
    key_int = int.from_bytes(key, byteorder='big')
    indices = np.arange(num_blocks)
    for i in range(num_blocks):
        swap_idx = (key_int + i) % num_blocks
        indices[i], indices[swap_idx] = indices[swap_idx], indices[i]
    return indices

# Step 3: Reverse the Block Shuffling using key K1
permutation = generate_deterministic_permutation(K1, num_blocks)
inverse_permutation = np.argsort(permutation)

# Unshuffle blocks based on the inverse permutation
final_blocks = [reversed_blocks[i] for i in inverse_permutation]


# Reconstruct blocks into rows
width = img_scrambled.shape[1] // block_size
rows = [cv.hconcat(final_blocks[i: i + width]) for i in range(0, len(final_blocks), width)]
img_reconstructed = cv.vconcat(rows)\

# Now proceed with splitting the channels and merging them back
height, width = img_reconstructed.shape
channel_width = width // 3

# Split the image into three separate channels
Y_channel = img_reconstructed[:, :channel_width]
Cb_channel = img_reconstructed[:, channel_width:2 * channel_width]
Cr_channel = img_reconstructed[:, 2 * channel_width:]


# Merge the channels back into a single YCbCr image
img_YCC_reconstructed = cv.merge([Y_channel, Cb_channel, Cr_channel])

# Convert from YCbCr back to RGB
img_final = cv.cvtColor(img_YCC_reconstructed, cv.COLOR_YCrCb2BGR)




# Display the reconstructed image
cv.imwrite("output/image_reconstructed.png", img_final)
cv.imshow("Reconstructed Image", img_final)
cv.waitKey(0)
cv.destroyAllWindows()





