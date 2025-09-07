# Handwritten Digit Recognition Service

This project demonstrates an end-to-end machine learning workflow for classifying handwritten digits from the MNIST dataset. It includes model training, an interactive web app, and an API.

---

## Features

- **Model Training:** Train a neural network on the MNIST dataset using TensorFlow & Keras.
- **Web App:** Test the model interactively using Streamlit.
- **API:** Serve predictions via FastAPI.
- **Preprocessing:** Handles image conversion, resizing, inversion, and normalization.

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/handwritten-digit-recognition.git
cd handwritten-digit-recognition
```

### 2. Set Up Python Environment

> **TensorFlow does not support Python 3.13+. Use Python 3.10 or 3.11.**

```bash
python --version  # Should show 3.10.x or 3.11.x
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt`, create one with:
```
streamlit
tensorflow
pillow
numpy
fastapi
uvicorn
```

### 4. Train the Model (Optional)

If you don't have `models/mnist_model.keras`, run the training script (provide your own or use the sample below):

```python
# train_model.py
import tensorflow as tf
from tensorflow import keras

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
model.save('models/mnist_model.keras')
```

### 5. Run the Streamlit App

```bash
streamlit run app.py
```

### 6. Run the FastAPI Service (Optional)

```bash
uvicorn api:app --reload
```

---

## Project Structure

```
handwritten-digit-recognition/
├── app.py
├── api.py                # (Optional: if you have a FastAPI API)
├── train_model.py        # (Script to train and save the model)
├── requirements.txt
├── models/
│   └── mnist_model.keras # (Trained model file, do NOT upload large files to GitHub)
├── resources/
│   ├── confusion_matrix.png
│   ├── accu_and_loss.png
│   └── streamlit-screen.png
├── .gitignore
└── README.md
```

---

## Screenshots

<img width="733" height="798" alt="image" src="https://github.com/user-attachments/assets/42db66b7-293a-481e-a04a-94c0fc71b7f1" />


---

## Notes

- Ensure the `models/mnist_model.keras` file exists before running the app or API.
- For best results, use clear, centered images of handwritten digits.
- If you encounter TensorFlow DLL errors, verify your Python version and install the [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170).

---


## License
