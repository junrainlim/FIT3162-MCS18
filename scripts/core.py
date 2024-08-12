import cv2 as cv
import numpy as np
import hashlib
import io


def encrypt(image_file):
    assert image_file is not None, "Image not found or unable to load."
    # Read the image file
    img = cv.imdecode(image_file, cv.IMREAD_UNCHANGED)

    # Convert from RGB to YCbCr
    img_YCC = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)

    # Combine YCbCr channels horizontally into a single grayscale image
    img_combined = cv.hconcat([img_YCC[:, :, 0], img_YCC[:, :, 1], img_YCC[:, :, 2]])

    # Define block size
    block_size = 8

    # Divide image into blocks of 8x8 pixels
    blocks = [
        img_combined[i : i + block_size, j : j + block_size]
        for i in range(0, img_combined.shape[0], block_size)
        for j in range(0, img_combined.shape[1], block_size)
    ]

    # Generate a complex key
    K1 = hashlib.sha256(b"secret_key_1").digest()
    K2 = hashlib.sha256(b"secret_key_2").digest()
    K3 = hashlib.sha256(b"secret_key_3").digest()

    # Use keys to seed the random number generators
    rng_1 = np.random.default_rng(int.from_bytes(K1, byteorder="big"))
    rng_2 = np.random.default_rng(int.from_bytes(K2, byteorder="big"))
    rng_3 = np.random.default_rng(int.from_bytes(K3, byteorder="big"))

    # Shuffle blocks positions based on the secret key K1
    rng_1.shuffle(blocks)

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

    def negative_positive_transformation(block, L=8):
        return 255 - block  # assuming 8-bit grayscale image

    # Step 2: Rotate and invert block using key K2
    transformed_blocks = []
    for block in blocks:
        angle = rng_2.choice([0, 90, 180, 270])  # Random rotation
        if rng_2.random() > 0.5:  # 0.5 value suggested from paper
            block = invert_block(block)  # Random inversion
        block = rotate_block(block, angle)
        transformed_blocks.append(block)

    # Step 3: Apply negative-positive transformation using K3
    final_blocks = []
    for block in transformed_blocks:
        if rng_3.random() > 0.5:
            block = negative_positive_transformation(block)
        final_blocks.append(block)

    # Reconstruct blocks into image
    width = img_combined.shape[1] // block_size
    rows = [
        cv.hconcat(final_blocks[i : i + width])
        for i in range(0, len(final_blocks), width)
    ]
    img_scrambled = cv.vconcat(rows)

    # Converting image to bytes for downloading
    img_bytes = io.BytesIO(cv.imencode(".jpg", img_scrambled)[1])
    img_bytes.seek(0)
    return img_bytes
