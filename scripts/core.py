import cv2 as cv
import numpy as np
import hashlib
import random
import io
import os

# Default block size in pixels
DEFAULT_BLOCK_SIZE = 8
# Default compression quality (0 = worst, 100 = best)
DEFAULT_COMPRESSION_QUALITY = 75

# Length of master key and salts used to derive additional keys from in bytes
MASTER_KEY_LENGTH = 1024
SALT_LENGTH = 16


def generate_key_triplet(
    secret_key: bytes, master_key_length: int, salt_length: int
) -> tuple[bytes, bytes, bytes, bytes]:
    """
    Given a secret key and the lengths its components use for the master key and salts in bytes, returns three keys which are derived from it.
    """

    def derive_key(master_key: bytes, salt: bytes) -> bytes:
        """
        Given a master key and salt, uses the PBKDF2 key derivation function to return a new, generated derived key in bytes. Uses 200000 iterations of SHA-256 as the hashing algorithm.
        """
        return hashlib.pbkdf2_hmac("sha256", master_key, salt, 200000)

    master_key = secret_key[:master_key_length]

    K1 = derive_key(master_key, secret_key[-salt_length * 3 :])
    K2 = derive_key(master_key, secret_key[-salt_length * 2 :])
    K3 = derive_key(master_key, secret_key[-salt_length * 1 :])
    return (K1, K2, K3)


def generate_index_sequence(key: bytes, length: int) -> np.ndarray:
    """
    Given a key and a length, returns a randomly permutated NumPy array of indices.
    """
    # Seed the PRNG with the key
    random.seed(key)

    # Generating array of indices
    indices = np.arange(length)

    # Shuffle the indices with controlled randomness
    for i in range(length):
        # -1 as end is inclusive
        swap_idx = random.randint(0, length - 1)
        # Swap items
        indices[i], indices[swap_idx] = indices[swap_idx], indices[i]

    return indices


def generate_rotation_sequence(key: bytes, length: int) -> np.ndarray:
    """
    Given a key and a length, returns a randomly permutated NumPy array of integers representing multiples of right angles that the element should be rotated clockwise by.

    0 represents a rotation by 0 degrees, 1 represents a rotation by 90, ...
    """
    # Seed the PRNG with the key
    random.seed(key)

    # Generating array of integers (8-bit signed int)
    angles = np.zeros(length, dtype=np.byte)

    for i in range(length):
        angles[i] = random.randint(0, 3)

    return angles


def generate_inversion_sequence(key: bytes, length: int) -> np.ndarray:
    """
    Given a key and a length, returns a randomly permutated NumPy array of booleans representing whether or not an element should be inverted.
    """
    # Seed the PRNG with the key
    random.seed(key)

    # Generating array of booleans
    inversions = np.zeros(length, dtype=bool)

    for i in range(length):
        # getrandbits is faster than choice
        inversions[i] = bool(random.getrandbits(1))

    return inversions


def generate_negative_positive_sequence(key: bytes, length: int) -> np.ndarray:
    """
    Given a key and a length, returns a randomly permutated NumPy array of booleans representing whether or not an negative positive transformation should be applied to an element.

    Implementation is the same as the one for generating an inversion sequence.
    """
    return generate_inversion_sequence(key, length)


# Block transformation functions
def rotate_block(block: np.ndarray, angle_index: int) -> np.ndarray:
    """
    Given a block and an angle index, returns the block rotated by the index multiplied by 90 degrees clockwise.
    """
    match angle_index:
        case 0:
            return block
        case 1:
            return cv.rotate(block, cv.ROTATE_90_CLOCKWISE)
        case 2:
            return cv.rotate(block, cv.ROTATE_180)
        case 3:
            return cv.rotate(block, cv.ROTATE_90_COUNTERCLOCKWISE)


def invert_block(block: np.ndarray) -> np.ndarray:
    """
    Given a block, returns the block inverted both horizontally and vertically.
    """
    return cv.flip(block, 1)


def image_to_blocks(image: np.ndarray, block_size: int) -> list[np.ndarray]:
    """
    Given a NumPy array representing an image and a block size in pixels, returns the image as a list of blocks of the given size.
    """
    return [
        image[i : i + block_size, j : j + block_size]
        for i in range(0, image.shape[0], block_size)
        for j in range(0, image.shape[1], block_size)
    ]


def blocks_to_image(blocks: list[np.ndarray], width: int) -> np.ndarray:
    """
    Given a list of blocks and the width of the original image in pixels, returns a NumPy array representing the reconstructed image.
    """
    rows = [cv.hconcat(blocks[i : i + width]) for i in range(0, len(blocks), width)]
    return cv.vconcat(rows)


def negative_positive_transform(block: np.ndarray, bits=8) -> np.ndarray:
    """
    Given a block, returns the block with all of its bits inverted. Assumes the image has 8-bits by default.
    """
    # Same as XOR-ing each pixel value in the block with 1 (from 2**bits)
    # e.g. 254 becomes 1, 152 becomes 130, ...
    return (2**bits) - 1 - block


def apply_block_scrambling(blocks: list[np.ndarray], key: bytes) -> list[np.ndarray]:
    """
    Given an list of blocks and a key, randomly scrambles the position of the blocks, returning the new list.
    """
    # Generating random sequence of indices
    indices = generate_index_sequence(key, len(blocks))

    # Applying block scrambling
    return [blocks[i] for i in indices]


def inverse_block_scrambling(blocks: list[np.ndarray], key: bytes) -> list[np.ndarray]:
    """
    Given an list of blocks and a key, applies the inverse of randomly scrambling the position of the blocks, returning the new list.

    The inverse of a permutation of a list is the list of indices that sort the permutation!
    """
    # Recovering random sequence of indices
    indices = generate_index_sequence(key, len(blocks))

    # Obtaining inverse of permutation
    # Algorithm obtained from https://stackoverflow.com/a/25535723. Should be faster than np.argsort for arrays of length N > 1210 (which should be the case for most images)
    # inverse_indices = np.argsort(indices)
    inverse_indices = np.zeros(len(indices), dtype=np.int32)
    i = np.arange(len(indices), dtype=np.int32)
    np.put(inverse_indices, indices, i)

    # Reversing block scrambling
    return [blocks[i] for i in inverse_indices]


def apply_rotation_inversion(blocks: list[np.ndarray], key: bytes) -> list[np.ndarray]:
    """
    Given an list of blocks and a key, randomly rotates and inverts blocks in the array, returning the new list.
    """
    # Generating random sequences of rotations and inversions
    rotations = generate_rotation_sequence(key, len(blocks))
    inversions = generate_inversion_sequence(key, len(blocks))

    new_blocks = []
    # Applying rotations and inversions
    for i, block in enumerate(blocks):
        angle_index = rotations[i]
        block = rotate_block(block, angle_index)
        if inversions[i]:
            block = invert_block(block)
        new_blocks.append(block)

    return new_blocks


def inverse_rotation_inversion(
    blocks: list[np.ndarray], key: bytes
) -> list[np.ndarray]:
    """
    Given an list of blocks and a key, applies the inverse of randomly rotating and inverting blocks in the array, returning the new list.

    The inverse of the rotation applied is obtained by converting the angle rotated to be counter-clockwise. The inverse of the inversion transformation is itself, and so is unchanged.
    """
    # Recovering random sequences of rotations and inversions
    rotations = generate_rotation_sequence(key, len(blocks))
    inversions = generate_inversion_sequence(key, len(blocks))

    new_blocks = []
    # Reversing rotations and inversions
    for i, block in enumerate(blocks):
        # Invert first, then rotate
        if inversions[i]:
            block = invert_block(block)
        angle_index = rotations[i]
        # Invert the angle index
        block = rotate_block(block, (4 - angle_index) % 4)

        new_blocks.append(block)

    return new_blocks


def apply_negative_positive_transform(
    blocks: list[np.ndarray], key: bytes
) -> list[np.ndarray]:
    """
    Given an list of blocks and a key, randomly applies a negative-positive transformation each pixel of the blocks in the array, returning the new list.
    """
    # Generating random sequence of negative positive transforms
    negative_positive_transforms = generate_negative_positive_sequence(key, len(blocks))

    new_blocks = []
    # Applying negative positive transforms
    for i, block in enumerate(blocks):
        if negative_positive_transforms[i]:
            block = negative_positive_transform(block)
        new_blocks.append(block)

    return new_blocks


def inverse_negative_positive_transform(
    blocks: list[np.ndarray], key: bytes
) -> list[np.ndarray]:
    """
    Given an list of blocks and a key, applies the inverse of a negative-positive transformation each pixel of the blocks in the array, returning the new list.

    Due to the nature of the negative positive transform, the inverse of this transformation is the transformation itself.
    """
    return apply_negative_positive_transform(blocks, key)


def encrypt(
    img: cv.Mat,
    block_size: int = DEFAULT_BLOCK_SIZE,
    compression_quality: int = DEFAULT_COMPRESSION_QUALITY,
) -> tuple[io.BytesIO, str]:
    """
    Given a NumPy array representing an image, returns an encrypted version of it in bytes, and the secret key used to encrypt it as a key.
    """
    # Generating secret key to be used for decryption
    # Using first 1024 bytes as master key, and every 16 bytes after for salts for derived keys(sizes for both as per hashlib suggestions)
    secret_key = os.urandom(MASTER_KEY_LENGTH + (SALT_LENGTH * 3))

    # Deriving individual keys for each step
    K1, K2, K3 = generate_key_triplet(secret_key, MASTER_KEY_LENGTH, SALT_LENGTH)
    # print("Encryption keys:\n1:", K1, "\n2:", K2, "\n3:", K3)

    # Convert from RGB to YCbCr
    img_YCC = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)

    # Combine YCbCr channels horizontally into a single grayscale image
    img_combined = cv.hconcat([img_YCC[:, :, 0], img_YCC[:, :, 1], img_YCC[:, :, 2]])
    print(type(img_combined))

    # Dividing image into blocks of 8x8 pixels
    blocks = image_to_blocks(img_combined, block_size)

    # Step 1: Scrambling blocks using K1
    blocks = apply_block_scrambling(blocks, K1)

    # Step 2: Rotating and inverting blocks using K2
    blocks = apply_rotation_inversion(blocks, K2)

    # Step 3: Applying negative-positive transformation using K3
    blocks = apply_negative_positive_transform(blocks, K3)

    # Reconstructing blocks into encrypted image
    width = img_combined.shape[1] // block_size
    encrypted_img = blocks_to_image(blocks, width)
    print("enrypt ", encrypted_img.shape)

    # Compressing then image then converting to bytes for downloading
    encode_parameters = [int(cv.IMWRITE_JPEG_QUALITY), compression_quality]
    encrypted_img_bytes = io.BytesIO(
        cv.imencode("100.jpg", encrypted_img, encode_parameters)[1]
    )
    encrypted_img_bytes.seek(0)

    return (encrypted_img_bytes, secret_key.hex())


def decrypt(
    img: cv.Mat, secret_key_hex: str, block_size: int = DEFAULT_BLOCK_SIZE
) -> bytes:
    """
    Given a NumPy array representing an image encrypted using this application, and a string of hexadecimal characters representing the secret key used to encrypt it, returns the decrypted image as bytes.
    """
    # Obtaining secret key and components from string
    secret_key = bytes.fromhex(secret_key_hex)
    K1, K2, K3 = generate_key_triplet(secret_key, MASTER_KEY_LENGTH, SALT_LENGTH)
    # print("Decryption keys:\n1:", K1, "\n2:", K2, "\n3:", K3)

    # Dividing image into blocks of 8x8 pixels
    blocks = image_to_blocks(img, block_size)

    # Step 1: Reversing negative-positive transformations using K3
    blocks = inverse_negative_positive_transform(blocks, K3)

    # Step 2: Reversing rotation and inversion of blocks using K2
    blocks = inverse_rotation_inversion(blocks, K2)

    # Step 3: Reversing block scrambling using K1
    blocks = inverse_block_scrambling(blocks, K1)

    # Reconstructing blocks into decrypted image
    width = img.shape[1] // block_size
    img_reconstructed = blocks_to_image(blocks, width)

    # Now proceed with splitting the channels and merging them back
    width = img_reconstructed.shape[1]
    channel_width = width // 3

    # Split the image into three separate channels
    Y_channel = img_reconstructed[:, :channel_width]
    Cb_channel = img_reconstructed[:, channel_width : 2 * channel_width]
    Cr_channel = img_reconstructed[:, 2 * channel_width :]

    # Merge the channels back into a single YCbCr image
    img_YCC_reconstructed = cv.merge([Y_channel, Cb_channel, Cr_channel])

    # Convert from YCbCr back to RGB
    decrypted_img = cv.cvtColor(img_YCC_reconstructed, cv.COLOR_YCrCb2BGR)

    # Converting image to bytes for downloading
    decrypted_img_bytes = io.BytesIO(cv.imencode(".jpg", decrypted_img)[1])
    decrypted_img_bytes.seek(0)

    return decrypted_img_bytes
