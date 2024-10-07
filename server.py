import numpy as np
import os
import cv2 as cv
from flask import (
    Flask,
    render_template,
    request,
    send_file,
)
from scripts.core import DEFAULT_BLOCK_SIZE, encrypt, decrypt

# Allowed file extensions (images only)
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "tif"}

app = Flask(__name__)
app.secret_key = os.urandom(16).hex()
app.config["SESSION TYPE"] = "filesystem"


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/encrypt", methods=["POST"])
def route_encrypt():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file selected!", 400
        file = request.files["file"]
        filename = file.filename
        # Reject empty files which are submitted when user does not select a file
        if filename == "":
            return "Please select a file to upload.", 400
        # Reject files with disallowed extensions
        if not (
            "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
        ):
            return "Please select a valid image format.", 415
        # Converting to NumPy array
        img_array = np.frombuffer(file.read(), np.int8)
        # Converting to OpenCV matrix
        img_mat = cv.imdecode(img_array, cv.IMREAD_UNCHANGED)

        # Encrypting image
        encrypted_image_bytes, secret_key = encrypt(img_mat)

        # Sending image file to user as an attachment
        response = send_file(
            encrypted_image_bytes,
            as_attachment=True,
            mimetype="image/jpg",
            download_name="encrypted_image.jpg",
        )
        # Including secret key as an additional header
        response.headers["key"] = secret_key
        return response


@app.route("/decrypt", methods=["POST"])
def route_decrypt():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file selected!", 400
        file = request.files["file"]
        filename = file.filename
        # Reject empty files which are submitted when user does not select a file
        if filename == "":
            return "Please select a file to upload.", 400
        # Reject files with disallowed extensions
        if not (
            "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
        ):
            return "Please select a valid image format.", 415
        # Converting to NumPy array
        img_array = np.frombuffer(file.read(), np.int8)
        # Converting to OpenCV matrix
        img_mat = cv.imdecode(img_array, cv.IMREAD_UNCHANGED)
        height, width = img_mat.shape
        # Checking if image width and height are multiples of block size
        if not (width % DEFAULT_BLOCK_SIZE == 0 and height % DEFAULT_BLOCK_SIZE == 0):
            return (
                "Please select an image which has a width and height which are multiples of "
                + str(DEFAULT_BLOCK_SIZE)
                + " pixels.",
                422,
            )
        if "key" not in request.form or not request.form["key"].strip():
            return "No key provided!", 400
        key = request.form["key"]
        # Decrypting image
        try:
            decrypted_image_bytes = decrypt(img_mat, key)
        except ValueError:
            return "Invalid secret key or error during decryption!", 400
        # Sending image file to user as an attachment
        response = send_file(
            decrypted_image_bytes,
            as_attachment=True,
            mimetype="image/jpg",
            download_name="decrypted_image.jpg",
        )
        return response


if __name__ == "__main__":
    app.run(debug=True)
