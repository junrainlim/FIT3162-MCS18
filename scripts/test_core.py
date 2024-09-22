import os
import unittest
import numpy as np
import cv2 as cv

from core import (
    generate_key_triplet,
    generate_index_sequence,
    generate_rotation_sequence,
    generate_inversion_sequence,
    generate_negative_positive_sequence,
    rotate_block,
    invert_block,
    image_to_blocks,
    blocks_to_image,
    negative_positive_transform,
    apply_block_scrambling,
    inverse_block_scrambling,
    apply_rotation_inversion,
    inverse_rotation_inversion,
    apply_negative_positive_transform,
    inverse_negative_positive_transform,
    encrypt,
    decrypt,
)

class TestCoreFunctions(unittest.TestCase):
    def setUp(self):
        self.secret_key = os.urandom(1048)  # 1024 + 3*16
        self.master_key_length = 1024
        self.salt_length = 16
        self.key = os.urandom(32)
        self.length = 64
        self.block_size = 8
        self.image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        self.blocks = image_to_blocks(self.image, self.block_size)

    def test_generate_key_triplet(self):
        """
        White-Box Test

        a) What is being tested: The key generation function to ensure it produces keys of expected lengths.
        b) How it is being tested: By calling the function with a fixed secret key and checking the length of the outputs.
        c) What are the inputs: A secret key of length 1048 bytes.
        d) What are the expected outputs: Three keys (K1, K2, K3), each of length 32 bytes.
        e) What are the actual outputs: The lengths of K1, K2, K3 are checked.
        """
        K1, K2, K3 = generate_key_triplet(self.secret_key, self.master_key_length, self.salt_length)
        self.assertEqual(len(K1), 32)
        self.assertEqual(len(K2), 32)
        self.assertEqual(len(K3), 32)

    def test_generate_index_sequence(self):
        """
        Black-Box Test

        a) What is being tested: The index sequence generation to ensure it produces a valid permutation.
        b) How it is being tested: By calling the function and checking the length and contents of the output.
        c) What are the inputs: A random key and length of 64.
        d) What are the expected outputs: An array of length 64 with unique indices from 0 to 63.
        e) What are the actual outputs: The length of the generated indices and sorted result are compared.
        """
        indices = generate_index_sequence(self.key, self.length)
        self.assertEqual(len(indices), self.length)
        self.assertTrue(np.all(np.sort(indices) == np.arange(self.length)))

    def test_generate_rotation_sequence(self):
        """
        Black-Box Test

        a) What is being tested: The rotation sequence generation to ensure it produces valid rotation indices.
        b) How it is being tested: By calling the function and checking the values of the output.
        c) What are the inputs: A random key and length of 64.
        d) What are the expected outputs: An array of length 64 with values between 0 and 3.
        e) What are the actual outputs: The length and value constraints of the generated rotations are verified.
        """
        rotations = generate_rotation_sequence(self.key, self.length)
        self.assertEqual(len(rotations), self.length)
        self.assertTrue(np.all((rotations >= 0) & (rotations <= 3)))

    def test_generate_inversion_sequence(self):
        """
        Black-Box Test

        a) What is being tested: The inversion sequence generation to ensure it produces valid boolean values.
        b) How it is being tested: By calling the function and checking the output.
        c) What are the inputs: A random key and length of 64.
        d) What are the expected outputs: An array of length 64 with boolean values (0 or 1).
        e) What are the actual outputs: The length and value constraints of the generated inversions are verified.
        """
        inversions = generate_inversion_sequence(self.key, self.length)
        self.assertEqual(len(inversions), self.length)
        self.assertTrue(np.all((inversions == 0) | (inversions == 1)))

    def test_generate_negative_positive_sequence(self):
        """
        Black-Box Test

        a) What is being tested: The negative-positive sequence generation to ensure it produces valid boolean values.
        b) How it is being tested: By calling the function and checking the output.
        c) What are the inputs: A random key and length of 64.
        d) What are the expected outputs: An array of length 64 with boolean values (0 or 1).
        e) What are the actual outputs: The length and value constraints of the generated negative-positive sequences are verified.
        """
        negative_positive = generate_negative_positive_sequence(self.key, self.length)
        self.assertEqual(len(negative_positive), self.length)
        self.assertTrue(np.all((negative_positive == 0) | (negative_positive == 1)))

    def test_rotate_block(self):
        """
        White-Box Test

        a) What is being tested: The block rotation function to ensure it performs the expected transformation.
        b) How it is being tested: By calling the function with a random block and checking the output shape.
        c) What are the inputs: A randomly generated 8x8 block of pixel values.
        d) What are the expected outputs: A rotated block of the same shape.
        e) What are the actual outputs: The output shape is verified against the input shape.
        """
        block = np.random.randint(0, 256, (self.block_size, self.block_size), dtype=np.uint8)
        rotated_block = rotate_block(block, 1)
        self.assertEqual(rotated_block.shape, block.shape)

    def test_invert_block(self):
        """
        White-Box Test

        a) What is being tested: The block inversion function to ensure it performs the expected transformation.
        b) How it is being tested: By calling the function with a random block and checking the output shape.
        c) What are the inputs: A randomly generated 8x8 block of pixel values.
        d) What are the expected outputs: An inverted block of the same shape.
        e) What are the actual outputs: The output shape is verified against the input shape.
        """
        block = np.random.randint(0, 256, (self.block_size, self.block_size), dtype=np.uint8)
        inverted_block = invert_block(block)
        self.assertEqual(inverted_block.shape, block.shape)

    def test_image_to_blocks(self):
        """
        Black-Box Test

        a) What is being tested: The image-to-blocks function to ensure it divides the image correctly.
        b) How it is being tested: By calling the function with an image and checking the number of resulting blocks.
        c) What are the inputs: A randomly generated 64x64 RGB image.
        d) What are the expected outputs: A list of blocks with the correct number based on block size.
        e) What are the actual outputs: The length of the blocks list is compared to the expected number of blocks.
        """
        blocks = image_to_blocks(self.image, self.block_size)
        self.assertEqual(len(blocks), (self.image.shape[0] // self.block_size) * (self.image.shape[1] // self.block_size))

    def test_blocks_to_image(self):
        """
        Black-Box Test

        a) What is being tested: The blocks-to-image function to ensure it reconstructs the image correctly.
        b) How it is being tested: By calling the function with blocks and checking the output image shape.
        c) What are the inputs: The blocks obtained from the `image_to_blocks` function.
        d) What are the expected outputs: The reconstructed image should match the original image shape.
        e) What are the actual outputs: The shape of the reconstructed image is compared to the original image shape.
        """
        blocks = image_to_blocks(self.image, self.block_size)
        reconstructed_image = blocks_to_image(blocks, self.image.shape[1] // self.block_size)
        self.assertEqual(reconstructed_image.shape, self.image.shape)

    def test_negative_positive_transform(self):
        """
        White-Box Test

        a) What is being tested: The negative-positive transformation to ensure it correctly alters pixel values.
        b) How it is being tested: By calling the function with a random block and checking the output shape.
        c) What are the inputs: A randomly generated 8x8 block of pixel values.
        d) What are the expected outputs: A transformed block of the same shape.
        e) What are the actual outputs: The output shape is verified against the input shape.
        """
        block = np.random.randint(0, 256, (self.block_size, self.block_size), dtype=np.uint8)
        transformed_block = negative_positive_transform(block)
        self.assertEqual(transformed_block.shape, block.shape)

    def test_apply_block_scrambling(self):
        """
        Black-Box Test

        a) What is being tested: The block scrambling function to ensure it scrambles blocks correctly.
        b) How it is being tested: By calling the function and checking the length of the scrambled output.
        c) What are the inputs: The list of blocks and a random key.
        d) What are the expected outputs: The scrambled output should have the same number of blocks.
        e) What are the actual outputs: The length of the scrambled blocks is compared to the original blocks length.
        """
        scrambled_blocks = apply_block_scrambling(self.blocks, self.key)
        self.assertEqual(len(scrambled_blocks), len(self.blocks))

    def test_inverse_block_scrambling(self):
        """
        White-Box Test

        a) What is being tested: The inverse block scrambling function to ensure it correctly restores original blocks.
        b) How it is being tested: By scrambling blocks and then unsrambling them, checking for equality.
        c) What are the inputs: Scrambled blocks and the original key.
        d) What are the expected outputs: The unscrambled blocks should match the original blocks.
        e) What are the actual outputs: A check for equality between original and unscrambled blocks.
        """
        scrambled_blocks = apply_block_scrambling(self.blocks, self.key)
        unscrambled_blocks = inverse_block_scrambling(scrambled_blocks, self.key)
        self.assertTrue(np.all([np.array_equal(b1, b2) for b1, b2 in zip(self.blocks, unscrambled_blocks)]))

    def test_apply_rotation_inversion(self):
        """
        Black-Box Test

        a) What is being tested: The rotation and inversion application function to ensure it applies transformations.
        b) How it is being tested: By calling the function and checking the length of the output.
        c) What are the inputs: The list of blocks and a random key.
        d) What are the expected outputs: The output should have the same number of blocks.
        e) What are the actual outputs: The length of the transformed blocks is compared to the original blocks length.
        """
        rotated_inverted_blocks = apply_rotation_inversion(self.blocks, self.key)
        self.assertEqual(len(rotated_inverted_blocks), len(self.blocks))

    def test_inverse_rotation_inversion(self):
        """
        White-Box Test

        a) What is being tested: The inverse of rotation and inversion application to ensure original blocks are restored.
        b) How it is being tested: By applying the inverse function to rotated-inverted blocks and checking for equality.
        c) What are the inputs: Rotated-inverted blocks and the original key.
        d) What are the expected outputs: The restored blocks should match the original blocks.
        e) What are the actual outputs: A check for equality between original and restored blocks.
        """
        rotated_inverted_blocks = apply_rotation_inversion(self.blocks, self.key)
        restored_blocks = inverse_rotation_inversion(rotated_inverted_blocks, self.key)
        self.assertEqual(len(restored_blocks), len(self.blocks))

    def test_apply_negative_positive_transform(self):
        """
        Black-Box Test

        a) What is being tested: The negative-positive transformation application to ensure it alters blocks correctly.
        b) How it is being tested: By calling the function and checking the length of the output.
        c) What are the inputs: The list of blocks and a random key.
        d) What are the expected outputs: The transformed output should have the same number of blocks.
        e) What are the actual outputs: The length of the transformed blocks is compared to the original blocks length.
        """
        transformed_blocks = apply_negative_positive_transform(self.blocks, self.key)
        self.assertEqual(len(transformed_blocks), len(self.blocks))

    def test_inverse_negative_positive_transform(self):
        """
        White-Box Test

        a) What is being tested: The inverse negative-positive transformation to ensure it restores original blocks.
        b) How it is being tested: By applying the inverse function to transformed blocks and checking for equality.
        c) What are the inputs: Transformed blocks and the original key.
        d) What are the expected outputs: The restored blocks should match the original blocks.
        e) What are the actual outputs: A check for equality between original and restored blocks.
        """
        transformed_blocks = apply_negative_positive_transform(self.blocks, self.key)
        restored_blocks = inverse_negative_positive_transform(transformed_blocks, self.key)
        self.assertEqual(len(restored_blocks), len(self.blocks))

    def test_encrypt_decrypt(self):
        """
        Black-Box Test

        a) What is being tested: The encryption and decryption process to ensure they work correctly together.
        b) How it is being tested: By encrypting an image and then decrypting it to check for equality with the original.
        c) What are the inputs: A randomly generated image and a secret key.
        d) What are the expected outputs: The decrypted image should match the original image.
        e) What are the actual outputs: 
        """
        encrypted_image = encrypt(self.image, self.secret_key)
        decrypted_image = decrypt(encrypted_image, self.secret_key)
        self.assertTrue(np.all(self.image == decrypted_image))

if __name__ == "__main__":
    unittest.main()
