# MNIST Digits Classification

A simple Artificial Neural Network (ANN) built with Keras 3 that classifies handwritten digits (0–9) from the MNIST dataset, and can predict digits from your own images.

## Project Files

| File | Description |
|---|---|
| `MINIST_Digits_Classification.ipynb` | Main notebook: data loading, preprocessing, model training, evaluation, and inference on a custom image |
| `digits_intel.h5` | Saved trained model (Keras H5 format) |

## Requirements

- Python 3.x
- keras (3.x)
- tensorflow (backend for Keras)
- numpy
- pandas
- matplotlib
- pillow (PIL)

Install with:
```bash
pip install keras tensorflow numpy pandas matplotlib pillow
```

## Dataset

Uses the built-in MNIST dataset via `keras.datasets.mnist`:
- Training set: 60,000 images, shape `(60000, 28, 28)`
- Test set: 10,000 images, shape `(10000, 28, 28)`
- Pixel values range 0–255 (grayscale), normalized to 0–1 before training

## Model Architecture

A fully-connected (Dense) ANN, not a CNN:

```
Input (28x28 image)
   → Flatten (784)
   → Dense(100, activation='relu')
   → Dense(100, activation='relu')
   → Dense(10, activation='softmax')   # 10 output classes (digits 0–9)
```

**Compilation:**
- Optimizer: `rmsprop`
- Loss: `sparse_categorical_crossentropy` (labels are integers, not one-hot)
- Metric: `sparse_categorical_accuracy`

**Training:**
- Batch size: 32
- Epochs: 10
- Validation data: MNIST test set

## How to Run

1. Open `MINIST_Digits_Classification.ipynb` in Jupyter Notebook / JupyterLab.
2. Run the cells in order:
   - **Load & explore data** — loads MNIST, checks shapes, visualizes a sample digit
   - **Preprocess** — normalizes pixel values to the 0–1 range
   - **Build & compile model** — defines the ANN architecture above
   - **Train** — fits the model for 10 epochs
   - **Evaluate / predict** — runs predictions on the test set
   - **Save model** — exports the trained model to `digits_intel.h5`
   - **Predict a custom image** — loads an external PNG/JPG, converts to grayscale, resizes to 28×28, and predicts the digit

## Predicting on Your Own Image

The notebook includes a `predict_image()` helper:

```python
from keras.models import load_model
digite_brain = load_model("digits_intel.h5")

def predict_image(image):
    image = image / 255.0
    image = np.expand_dims(a=image, axis=0)  # add batch dimension -> (1, 28, 28)
    prediction = digite_brain.predict(image)
    return np.argmax(prediction)
```

To predict your own digit image:
1. Update the `img_path` variable to point to your image file.
2. Convert it to grayscale and resize to 28×28 using PIL:
   ```python
   from PIL import Image
   import numpy as np

   image = Image.open(img_path).convert(mode='L').resize(size=(28, 28))
   image = np.array(image)
   predict_image(image)
   ```
3. The function returns the predicted digit (0–9) as the model's argmax output.

> **Note:** The current `img_path` in the notebook is a local Windows path — update it to your own image location before running.

## Notes / Known Quirks

- This is a plain ANN (Dense layers only), not a CNN — accuracy will be lower than a convolutional model, though usually still >97% on MNIST.
- `predict_image()` is defined *after* it's first called in the notebook's current cell order — if running top-to-bottom, define the function before its first use.
- `.h5` is used as an intermediate/legacy save format for the trained Keras model.

## Author

Tamilarasan K
