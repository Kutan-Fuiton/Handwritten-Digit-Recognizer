# run_model just run the model on the test dataset 

import keras
import matplotlib
matplotlib.use('Agg') 
from keras.datasets import mnist
from keras.models import load_model
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

# Load trained model
model = load_model("model/mnist_cnn.h5")
print("✅ Model loaded successfully!")

# Load MNIST test dataset
(_, _), (x_test, y_test) = mnist.load_data()

# Preprocess test data
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
y_test = to_categorical(y_test, 10)

# Evaluate model performance
loss, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {acc * 100:.2f}%")

# Pick a sample image
sample_index = 12  
sample_image = x_test[sample_index].reshape(1, 28, 28, 1)

# Predict
prediction = model.predict(sample_image)
predicted_digit = np.argmax(prediction)
print(f"Predicted Digit: {predicted_digit}")

plt.imshow(x_test[sample_index].reshape(28, 28), cmap="gray")
plt.title(f"Prediction: {predicted_digit}")
plt.axis("off")
# Save the visualization to a file
plt.savefig("predicted_digit.png")
print("Image saved as predicted_digit.png")
