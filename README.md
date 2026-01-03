# ✍️ Handwritten Digit Recognizer (MNIST CNN)

A Deep Learning project that recognizes handwritten digits (0-9) using a Convolutional Neural Network (CNN) trained on the MNIST dataset.

## 🚀 Project Evolution: From FastAPI to Streamlit

This project started as a full-stack application with a **FastAPI** backend and a custom **HTML/CSS/JS** frontend. 

However, to optimize for deployment and ease of interaction, the project was migrated to **Streamlit**.

### What is Streamlit?
Streamlit is an open-source Python framework that allows data scientists to turn data scripts into shareable web apps in minutes. It removes the need for separate frontend code, allowing the logic and UI to exist in a single, deployable Python script.

## 🧠 Model & Architecture

The core of this project is a **CNN (Convolutional Neural Network)** built with TensorFlow/Keras.

### The Model (`model/mnist_cnn.h5`)
- **Input:** 28x28 grayscale images.
- **Architecture:** - 2x Convolutional Layers (Relu activation)
  - 2x Max Pooling Layers
  - Flatten Layer
  - Dense Layers (Output: Softmax 10 classes)
- **Accuracy:** ~99% on test data.

### Smart Preprocessing
Unlike simple image resizing, this app mimics the original MNIST data processing pipeline to ensure high accuracy:
1. **Grayscale conversion** & Inversion (White text on Black background).
2. **Noise Removal** (Thresholding).
3. **Smart Cropping:** Finds the bounding box of the digit.
4. **Resizing:** Resizes the digit to 20x20px.
5. **Padding:** Centers the digit in a 28x28px canvas with a 4px padding border (Center of Mass).

## 🛠️ Tech Stack

- **Python 3.x**
- **TensorFlow / Keras** (Model Training & Inference)
- **Streamlit** (Web Interface)
- **OpenCV** (Image Processing)
- **NumPy** (Matrix operations)

## 📂 Project Structure

```bash
├── model/
│   ├── mnist_cnn.h5       # Pre-trained CNN model
│   └── train_model.py     # Script used to train the model
├── streamlit_app.py       # Main application file
├── requirements.txt       # Project dependencies
├── runtime.txt            # streamlite python version
└── README.md              # Documentation
```

## 💻 How to Run Locally

To run this project on your local machine, open your terminal and execute the following bash commands:

### 1. Clone the repository
```bash
git clone https://github.com/Kutan-Fuiton/Handwritten-Digit-Recognizer
cd Handwritten-Digit-Recognizer
```

### 2. Create a virtual environment (Recommended)
```bash
# Windows:
python -m venv venv
venv\Scripts\activate
# Mac/Linux:
# python3 -m venv venv
# source venv/bin/activate
```

### 3. Install the required dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application
```bash
streamlit run streamlit_app.py
```

Once the command runs, the application will automatically open in your default web browser at:
http://localhost:8501
