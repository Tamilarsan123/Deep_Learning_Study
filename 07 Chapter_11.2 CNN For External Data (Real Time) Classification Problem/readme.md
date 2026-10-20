# CNN for External Data: Real-Time Image Classification

A Convolutional Neural Network (CNN) project that is trained on a labelled image dataset and then used to classify **new, external images and live camera frames in real time**.

> Part of my [Deep Learning Study](https://github.com/Tamilarasan123/Deep_Learning_Study) series, Chapter 11.2.

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Workflow](#workflow)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Future Improvements](#future-improvements)
- [Author](#author)

---

## Overview

Most CNN tutorials stop at evaluating on a held-out test split. This project goes one step further: it takes the trained model and applies it to **external data it has never seen**, including real-time input, to check how well it generalizes outside the training set.

**Problem type:** Multi-class image classification
**Goal:** Train a CNN, save it, and use it to predict classes for unseen images and live input

---

## Key Features

- End-to-end pipeline: data loading, preprocessing, training, evaluation, inference
- Image preprocessing and normalization
- Custom CNN built with Conv2D, pooling, dropout and dense layers
- Model saving and reloading for reuse
- Prediction on external images (any image outside the dataset)
- Real-time classification using a webcam feed
- Prediction with class label and confidence score

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.x |
| Deep Learning | TensorFlow / Keras |
| Computer Vision | OpenCV |
| Data Handling | NumPy, Pandas |
| Visualization | Matplotlib, Seaborn |
| Environment | Jupyter Notebook |

---

## Project Structure

```
07 Chapter_11.2 CNN For External Data (Real Time) Classification Problem/
│
├── dataset/                 # Training / validation images (class-wise folders)
├── notebook.ipynb           # Training, evaluation and prediction workflow
├── model/                   # Saved trained model (.h5 / .keras)
├── test_images/             # External images used for prediction
└── README.md
```

> Update file and folder names to match your repository.

---

## Dataset

- **Source:** [Add dataset name / link]
- **Classes:** [List your classes, e.g. class_1, class_2, class_3]
- **Image size:** [e.g. 64 x 64 / 128 x 128]
- **Split:** [e.g. 80% train / 20% validation]

---

## Model Architecture

```
Input (H x W x 3)
   ↓
Conv2D + ReLU → MaxPooling
   ↓
Conv2D + ReLU → MaxPooling
   ↓
Conv2D + ReLU → MaxPooling
   ↓
Flatten
   ↓
Dense + ReLU → Dropout
   ↓
Dense + Softmax (number of classes)
```

| Setting | Value |
|---|---|
| Optimizer | Adam |
| Loss | Categorical Crossentropy |
| Metrics | Accuracy |
| Epochs | [add] |
| Batch size | [add] |

---

## Workflow

1. **Load data** from class-wise folders
2. **Preprocess**: resize, normalize, augment (optional)
3. **Build** the CNN model
4. **Train** and validate
5. **Evaluate** with accuracy, loss curves, confusion matrix, classification report
6. **Save** the trained model
7. **Predict** on external images
8. **Real-time inference** using webcam frames

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Tamilarasan123/Deep_Learning_Study.git
cd "Deep_Learning_Study/07 Chapter_11.2 CNN For External Data (Real Time) Classification Problem"

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install tensorflow opencv-python numpy pandas matplotlib seaborn scikit-learn jupyter
```

---

## Usage

### 1. Train the model

Open the notebook and run all cells:

```bash
jupyter notebook
```

### 2. Predict on an external image

```python
import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("model/cnn_model.h5")
class_names = ["class_1", "class_2", "class_3"]   # update

img = cv2.imread("test_images/sample.jpg")
img = cv2.resize(img, (128, 128)) / 255.0
img = np.expand_dims(img, axis=0)

pred = model.predict(img)
print(class_names[np.argmax(pred)], f"{np.max(pred) * 100:.2f}%")
```

### 3. Real-time webcam classification

```python
import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("model/cnn_model.h5")
class_names = ["class_1", "class_2", "class_3"]   # update

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (128, 128)) / 255.0
    pred = model.predict(np.expand_dims(img, axis=0), verbose=0)
    label = class_names[np.argmax(pred)]
    conf = np.max(pred) * 100

    cv2.putText(frame, f"{label} ({conf:.1f}%)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Real-Time Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
```

Press **Q** to quit the webcam window.

---

## Results

| Metric | Value |
|---|---|
| Training accuracy | [add]% |
| Validation accuracy | [add]% |
| Test accuracy | [add]% |

Add screenshots here:

- Accuracy / loss curves
- Confusion matrix
- Sample predictions on external images
- Real-time webcam demo (GIF)

---

## Future Improvements

- Transfer learning (MobileNetV2, ResNet50, EfficientNet) for higher accuracy
- Stronger data augmentation to improve generalization
- Convert to TensorFlow Lite for faster real-time inference
- Deploy as a web app using Streamlit or Flask
- Add Grad-CAM to visualize what the model focuses on

---

## Author

**Tamilarasan K**
B.Tech, Artificial Intelligence & Data Science

- GitHub: [Tamilarasan123](https://github.com/Tamilarasan123)
- LinkedIn: [tamilarasan-k](https://linkedin.com/in/tamilarasan-k-03a27834b)

---

If you found this useful, consider giving the repository a star.
