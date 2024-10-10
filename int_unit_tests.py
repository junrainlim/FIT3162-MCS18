import unittest
from server import app
import os

class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()

    def test_combined_workflow(self):
    # Step 1: Encrypt the test image
        with open('test_image.jpg', 'rb') as img:
            original_image_size = os.path.getsize('test_image.jpg')  # Get size of original image
            print(f"Original image size: {original_image_size} bytes")
            encrypt_response = self.app.post('/encrypt', data={'file': img})
            self.assertEqual(encrypt_response.status_code, 200)
            self.assertIn('encrypted_image.jpg', encrypt_response.headers['Content-Disposition'])

        # Retrieve the secret key from the encryption response headers
            secret_key = encrypt_response.headers['key']

        # Step 2: Use the encrypted image to test decryption
            encrypted_img = encrypt_response.data  # Get the encrypted image from the response
            compressed_image_size = len(encrypted_img)    # Get size of the compressed, encrypted image
            print(f"Compressed (encrypted) image size: {compressed_image_size} bytes")

        # Assert that the compressed image size is smaller than the original image
            self.assertLess(compressed_image_size, original_image_size)

        # Convert the encrypted image to a file-like object
            from io import BytesIO
            encrypted_file = BytesIO(encrypted_img)
            encrypted_file.name = 'compressed_encrypted_image.jpg'

        # Send the encrypted file and the key for decryption
            decrypt_response = self.app.post(
                '/decrypt',
                data={
                'key': secret_key,
                'file': (encrypted_file, 'compressed_encrypted_image.jpg')  # File sent as part of the data
            },
                content_type='multipart/form-data'
            )
            self.assertEqual(decrypt_response.status_code, 200)
            self.assertIn('decrypted_image.jpg', decrypt_response.headers['Content-Disposition'])


class TestInvalidFileUpload(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()

    def test_invalid_file_upload(self):
        # print("Testing invalid file upload...")

        # Step 1: Simulate uploading a non-image file (e.g., .txt file)
        with open('test_file.txt', 'rb') as txt_file:
            response = self.app.post('/encrypt', data={'file': txt_file})

            # Step 2: Verify that the response status code is 415 (Unsupported Media Type)
            self.assertEqual(response.status_code, 415)

            # Optional: Check if the error message contains a meaningful message
            self.assertIn(b'Please select a valid image format.', response.data)
            # print(f"Response data: {response.data.decode()}")  # Output the response data for inspection


if __name__ == "__main__":
    unittest.main()
