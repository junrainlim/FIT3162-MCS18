import numpy as np
from flask import Flask, render_template, request, redirect, flash, url_for, send_file
from scripts.core import encrypt

# Allowed file extensions (images only)
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "tif"}

app = Flask(__name__)
app.secret_key = "SECRET KEY"
app.config["SESSION TYPE"] = "filesystem"


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


# Upload image for encryption
@app.route("/upload", methods=["POST"])
def upload():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file selected!", "error")
            return redirect(url_for("index"))
        file = request.files["file"]
        filename = file.filename
        # Reject empty files which are submitted when user does not select a file
        if filename == "":
            flash("Please select a file to upload.", "error")
            return redirect(url_for("index"))
        # Reject files with disallowed extensions
        if not (
            "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
        ):
            flash("Please select a valid image format.", "error")
            return redirect(url_for("index"))
        # Converting to NumPy array
        img_array = np.frombuffer(file.read(), np.int8)
        # Encrypting image
        encrypted_image = encrypt(img_array)
        # Sending image file to user
        return send_file(encrypted_image, as_attachment=True, mimetype="image/jpg", download_name="encrypted_image.jpg")


if __name__ == "__main__":
    app.run(debug=True)
