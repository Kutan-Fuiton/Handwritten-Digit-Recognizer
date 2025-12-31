# preprocess_image.py run the model on a custom image with the pre processing the image with the help of open cv

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.models import load_model

# confirming trained model exists
model = load_model("model/mnist_cnn.h5")
print("Model loaded successfully!")

# Function to preprocess image
def preprocess_image(path):

    # read image in grayscale which is basically a 2D array help for mnist set to process
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    # If average color is more than 127, invert the image (white digit on black bg)
    if np.mean(img) > 127:
        img = cv2.bitwise_not(img)

    # Threshold the image, like a color > 128 -> white(255), else black(0)
    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

    # For stroke adjusting
    kernel = np.ones((2, 2), np.uint8)  
    img = cv2.dilate(img, kernel, iterations=1)  

    # perfectly finding bounding box arround the digit
    coords = cv2.findNonZero(img)
    x, y, w, h = cv2.boundingRect(coords)
    digit = img[y:y+h, x:x+w]

    digit_resized = cv2.resize(digit, (20, 20), interpolation=cv2.INTER_AREA)
    padded = np.pad(digit_resized, ((4,4),(4,4)), "constant", constant_values=0)

    processed = padded.astype("float32") / 255.0
    processed = processed.reshape(1, 28, 28, 1)

    return processed, padded

# Example usage
img_array, display_img = preprocess_image("model/images/image.png")

# Predict
prediction = model.predict(img_array)
predicted_digit = np.argmax(prediction)
print(f"Predicted Digit: {predicted_digit}")

# Show the processed image
plt.imshow(display_img, cmap="gray")
plt.title(f"Prediction: {predicted_digit}")
plt.axis("off")
plt.savefig("model/predicted_digit.png")
