import streamlit as st
import numpy as np
import cv2
# from keras.models import load_model
from tensorflow import keras
from streamlit_drawable_canvas import st_canvas

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Handwritten Digit Recognizer",
    layout="centered"
)

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_cnn():
    return keras.models.load_model(
        "model/mnist_cnn.h5",
        compile=False
    )

model = load_cnn()

# ------------------ PREPROCESS (UNCHANGED LOGIC) ------------------
def preprocess_image(img):
    # Convert RGBA → RGB if needed
    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if np.mean(img) > 127:
        img = cv2.bitwise_not(img)

    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

    kernel = np.ones((2, 2), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)

    coords = cv2.findNonZero(img)
    if coords is None:
        return None

    x, y, w, h = cv2.boundingRect(coords)
    digit = img[y:y+h, x:x+w]

    digit_resized = cv2.resize(digit, (20, 20), interpolation=cv2.INTER_AREA)
    padded = np.pad(digit_resized, ((4, 4), (4, 4)), "constant", constant_values=0)

    processed = padded.astype("float32") / 255.0
    processed = processed.reshape(1, 28, 28, 1)

    return processed

# ------------------ UI ------------------
st.title("✍️ Handwritten Digit Recognizer")
st.write("Draw a digit (0–9) inside the box")

canvas = st_canvas(
    fill_color="rgba(0, 0, 0, 0)",
    stroke_width=15,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Predict"):
    if canvas.image_data is not None:
        img = canvas.image_data.astype(np.uint8)

        processed = preprocess_image(img)

        if processed is not None:
            prediction = model.predict(processed)
            digit = int(np.argmax(prediction))
            confidence = float(np.max(prediction)) * 100

            st.success(f"🧠 Predicted Digit: **{digit}**")
            st.info(f"Confidence: **{confidence:.2f}%**")
        else:
            st.warning("Please draw a digit first!")

st.caption("CNN trained on MNIST • Deployed using Streamlit")

