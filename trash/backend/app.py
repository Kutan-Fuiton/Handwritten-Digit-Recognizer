from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import cv2
import base64
from keras.models import load_model
import os

app = Flask(__name__, static_folder="../frontend", static_url_path="")
CORS(app)

# Load model
model = load_model("model/mnist_cnn.h5")

def preprocess_image(img):
    # Convert to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Invert if white digit on black
    if np.mean(img) > 127:
        img = cv2.bitwise_not(img)

    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)

    coords = cv2.findNonZero(img)
    x, y, w, h = cv2.boundingRect(coords)
    digit = img[y:y+h, x:x+w]

    digit_resized = cv2.resize(digit, (20, 20), interpolation=cv2.INTER_AREA)
    padded = np.pad(digit_resized, ((4, 4), (4, 4)), "constant", constant_values=0)

    processed = padded.astype("float32") / 255.0
    processed = processed.reshape(1, 28, 28, 1)

    return processed

@app.route("/")
def index():
    return send_from_directory("../frontend", "index.html")

@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory("../frontend", path)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json["image"]
        img_bytes = base64.b64decode(data.split(",")[1])
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        processed = preprocess_image(img)
        prediction = model.predict(processed)
        digit = int(np.argmax(prediction))

        return jsonify({"prediction": digit})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)