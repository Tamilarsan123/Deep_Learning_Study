# CIFAR-10 Image Classification (ANN)

A fully connected neural network (ANN) built with Keras to classify 32×32 color images from the CIFAR-10 dataset into 10 classes.

## Dataset

[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) — loaded directly via `keras.datasets.cifar10`.

| Split | Images | Shape |
|-------|--------|-------|
| Train | 50,000 | (32, 32, 3) |
| Test  | 10,000 | (32, 32, 3) |

**Classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## Project Structure

```
├── CIFAR10_Classification.ipynb   # Full workflow: data → model → evaluation → prediction
├── CIFAR10.h5                     # Trained model (saved weights + architecture)
└── README.md
```

## Workflow

1. **Import libraries** — Keras, NumPy, Pandas, Matplotlib
2. **Data collection** — load CIFAR-10 via Keras
3. **Data understanding** — verify shapes, visualize a sample image
4. **Data preparation** — normalize pixels from 0–255 to 0–1; define class names
5. **Model building** — Sequential ANN
6. **Training** — 20 epochs, batch size 64, 20% validation split
7. **Evaluation** — accuracy on the 10,000 test images
8. **Manual prediction** — predict a single test image and display actual vs. predicted label
9. **Save model** — exported to `CIFAR10.h5`

## Model Architecture

| Layer | Type | Units | Activation |
|-------|------|-------|------------|
| 1 | Flatten | 3072 | — |
| 2 | Dense | 512 | ReLU |
| 3 | Dense | 256 | ReLU |
| 4 | Dense | 10 | Softmax |

- **Optimizer:** Adam
- **Loss:** Sparse Categorical Crossentropy
- **Metric:** Sparse Categorical Accuracy
- **Trainable parameters:** 1,707,274

## Results

| Metric | Value |
|--------|-------|
| Final validation accuracy (epoch 20) | ~50.4% |
| **Test accuracy** | **~50.2%** |
| Test loss | ~1.43 |

Training accuracy reached ~57%, while validation/test accuracy stayed near 50%, so the model is overfitting and is limited by the ANN architecture. Flattening the images discards spatial structure, which a CNN preserves.

**Sample prediction:** test image #100 was a *deer* but predicted as *dog* (32.7% confidence) — the model confuses visually similar animal classes.

## Requirements

- Python 3.9+
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib

```bash
pip install tensorflow numpy pandas matplotlib
```

## Usage

**Run the notebook**

```bash
jupyter notebook CIFAR10_Classification.ipynb
```

**Load the trained model and predict**

```python
import numpy as np
import keras
import matplotlib.pyplot as plt

model = keras.models.load_model("CIFAR10.h5")

class_names = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

(_, _), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_test = x_test / 255.0

image = x_test[100]
pred = model.predict(np.expand_dims(image, axis=0))
print("Predicted:", class_names[np.argmax(pred)])
print("Actual:   ", class_names[y_test[100][0]])
```

## Future Improvements

- Replace the ANN with a **CNN** (Conv2D + MaxPooling) — typically 70–80%+ accuracy on CIFAR-10
- Add **Dropout** and **Batch Normalization** to reduce overfitting
- Apply **data augmentation** (flips, rotations, shifts)
- Add **EarlyStopping** and learning-rate scheduling
- Save in the native `.keras` format instead of legacy `.h5`
- Add a confusion matrix and per-class metrics

## Author

**Tamilarasan K**
[GitHub](https://github.com/Tamilarasan123) · [LinkedIn](https://linkedin.com/in/tamilarasan-k-03a27834b)
