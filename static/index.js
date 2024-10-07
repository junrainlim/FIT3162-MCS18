// Displays an error message in the HTML element.
const showError = (error) => {
  var errorDisplay = document.getElementById("error-display");
  errorDisplay.innerText = "Error: " + error;
  errorDisplay.hidden = false;
};

// Encrypts an image and starts the file download.
const encrypt = async () => {
  let data = new FormData();
  data.append("file", document.getElementById("encrypt-image-input").files[0]);
  const response = await fetch("/encrypt", {
    method: "POST",
    body: data,
  })
    .then((response) => {
      // Checking if the encryption was successful
      if (!response.ok) {
        // Extracting the response text to show as an error
        response.text().then((text) => {
          showError(text);
        });
        throw new Error();
      }
      console.log(response);
      // Displaying the secret key
      var keyText = document.getElementById("encrypt-key-output");
      keyText.value = response.headers.get("key");
      return response.blob();
    })
    .then((blob) => {
      // Temporarily creating an element to download the image
      let tempUrl = URL.createObjectURL(blob);
      const aTag = document.createElement("a");
      aTag.href = tempUrl;
      aTag.download = tempUrl.replace(/^.*[\\/]/, "") + ".jpg";
      document.body.appendChild(aTag);
      aTag.click();
      URL.revokeObjectURL(tempUrl);
      aTag.remove();
      // Hiding error display
      var errorDisplay = document.getElementById("error-display");
      errorDisplay.hidden = true;
    })
    .catch((error) => {});
};

// Copies the content of the key text box into the user's clipboard.
const copyKey = () => {
  var keyText = document.getElementById("encrypt-key-output");
  keyText.select();
  keyText.setSelectionRange(0, 99999);

  navigator.clipboard.writeText(keyText.value);
};

const decrypt = async () => {
  let data = new FormData();
  data.append("file", document.getElementById("decrypt-image-input").files[0]);
  data.append("key", document.getElementById("decrypt-key-input").value);
  const response = await fetch("/decrypt", {
    method: "POST",
    body: data,
  })
    .then((response) => {
      // Checking if the decryption was successful
      if (!response.ok) {
        // Extracting the response text to show as an error
        response.text().then((text) => {
          showError(text);
        });
        throw new Error();
      }
      console.log(response);
      return response.blob();
    })
    .then((blob) => {
      // Temporarily creating an element to download the image
      let tempUrl = URL.createObjectURL(blob);
      const aTag = document.createElement("a");
      aTag.href = tempUrl;
      aTag.download = tempUrl.replace(/^.*[\\/]/, "") + ".jpg";
      document.body.appendChild(aTag);
      aTag.click();
      URL.revokeObjectURL(tempUrl);
      aTag.remove();
      // Hiding error display
      var errorDisplay = document.getElementById("error-display");
      errorDisplay.hidden = true;
    })
    .catch((error) => {
      console.log(error);
    });
};

const previewImage = (event) => {
  var imagePreview = document.getElementById("image-preview");
  imagePreview.src = URL.createObjectURL(event.target.files[0]);
  imagePreview.onload = () => {
    URL.revokeObjectURL(imagePreview.src);
  };
};
