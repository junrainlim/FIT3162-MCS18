import numpy as np
from flask import Flask, render_template, request, redirect, flash, url_for, send_file
from scripts.core import encrypt as p_encrypt

# Name of folder where uploaded images are stored
UPLOAD_FOLDER = "uploads"
# Allowed file extensions (images only)
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "tif"}

app = Flask(__name__)
app.secret_key = "SECRET KEY"
app.config["SESSION TYPE"] = "filesystem"


@app.route("/", methods=["GET"])
def index():
    global uploaded_image
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
        # Global variable to keep image
        global uploaded_image
        # Converting to NumPy array
        img_array = np.frombuffer(file.read(), np.int8)
        # Updating global variable
        uploaded_image = img_array
        flash('Loaded image file "' + filename + '".', "success")
        # Save the file to the uploads folder
        # file.save(
        #     os.path.join(
        #         os.path.dirname(__file__),
        #         UPLOAD_FOLDER,
        #         # Extracting file extension to save
        #         ("uploaded_image" + os.path.splitext(file.filename)[1]),
        #     )
        # )
    return render_template("index.html")


# Encrypt image
@app.route("/encrypt", methods=["GET"])
def encrypt():
    try:
        uploaded_image
    except NameError:
        flash("Please upload an image first.", "error")
        return redirect(url_for("index"))
    else:
        p_encrypt(uploaded_image)
        return send_file("output/encrypted_image.jpg", as_attachment=True)
        # return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
