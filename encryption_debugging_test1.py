import cv2 as cv
import numpy as np
import hashlib
import random
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend

# Read the image
img_path = "images/nando-jpeg-quality-001.jpg"
img = cv.imread(img_path, cv.IMREAD_COLOR)
assert img is not None, "Image not found or unable to load."

# Convert from RGB to YCbCr
img_YCC = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)

# Combine YCbCr channels horizontally into a single grayscale image
img_combined = cv.hconcat([img_YCC[:, :, 0], img_YCC[:, :, 1], img_YCC[:, :, 2]])

# Define block size
block_size = 8

# Divide image into blocks of 8x8 pixels
blocks = [
    img_combined[i: i + block_size, j: j + block_size]
    for i in range(0, img_combined.shape[0], block_size)
    for j in range(0, img_combined.shape[1], block_size)
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


def generate_less_deterministic_permutation(key, num_blocks):
    # Seed the PRNG with the key
    key_int = int.from_bytes(key, byteorder='big')
    random.seed(key_int)

    # Generate a list of indices
    indices = np.arange(num_blocks)

    # Shuffle the indices with controlled randomness
    for i in range(num_blocks):
        swap_idx = random.randint(0, num_blocks - 1)
        indices[i], indices[swap_idx] = indices[swap_idx], indices[i]

    return indices


# determine key space and generate a fixed permutation for shuffling
num_blocks = len(blocks)
permutation = generate_less_deterministic_permutation(K1, num_blocks)

# Shuffle blocks based on the deterministic permutation
blocks = [blocks[i] for i in permutation]


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


# Create rotation sequence
rotation_sequence = generate_rotation_angles(K2, num_blocks)

# Create inversion sequence
inversion_sequence = generate_inversion_flags(K2, num_blocks)

# Step 2: Rotate and Invert blocks using key K2
transformed_blocks = []
for i, block in enumerate(blocks):
    angle = rotation_sequence[i]
    if inversion_sequence[i]:
        block = invert_block(block)
    block = rotate_block(block, angle)
    transformed_blocks.append(block)

# Get length of transformed blocks
num_blocks = len(transformed_blocks)


# Negative positive transformation function
def negative_positive_transformation(block, L=8):
    return 255 - block  # assuming 8-bit grayscale image


# Function to generate a deterministic sequence for negative positive transformation
def generate_negative_positive_flags(key: int, num_blocks: int) -> np.ndarray:
    key_int = int.from_bytes(key, byteorder='big')
    negative_positive_sequence = np.zeros(num_blocks, dtype=bool)
    for i in range(num_blocks):
        negative_positive_sequence[i] = (key_int + i) % 2 == 0
    return negative_positive_sequence


# Create negative positive sequence
negative_positive_sequence = generate_negative_positive_flags(K3, num_blocks)

# Step 3: Apply negative-positive transformation using K3
final_blocks = []
for i, block in enumerate(transformed_blocks):
    if negative_positive_sequence[i]:
        block = negative_positive_transformation(block)
    final_blocks.append(block)

# Reconstruct blocks into image
width = img_combined.shape[1] // block_size
rows = [cv.hconcat(final_blocks[i: i + width]) for i in range(0, len(final_blocks), width)]
img_scrambled = cv.vconcat(rows)

# Save the scrambled image
cv.imwrite("images/image_scrambled.png", img_scrambled)

# compression
cv.imwrite("compressed.jpg", img_scrambled, [cv.IMWRITE_JPEG_QUALITY, 100])

# Display the reconstructed image
cv.imshow("Display Window", img_scrambled)
cv.waitKey(0)
cv.destroyAllWindows()










