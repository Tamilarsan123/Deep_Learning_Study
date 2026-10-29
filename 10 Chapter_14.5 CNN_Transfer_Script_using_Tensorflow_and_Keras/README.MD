# CNN Transfer Learning using TensorFlow and Keras

Transfer learning with pretrained CNN models using TensorFlow and Keras. This is Chapter 14.5 of my deep learning study series.

## Overview
This project reuses pretrained CNN architectures for image classification. It loads a model trained on ImageNet, freezes its base layers, adds a custom classifier head, and fine-tunes it on a custom dataset.

## Features
- Loads pretrained models from `tf.keras.applications` [e.g., VGG16, ResNet50, MobileNetV2]
- Freezes the base layers and adds a custom classification head
- Image preprocessing and data augmentation
- Training with callbacks [EarlyStopping, ModelCheckpoint, if used]
- Fine-tuning of top layers [if included]
- Accuracy and loss plots

## Requirements
- Python 3.8+
- TensorFlow 2.x
- NumPy
- Matplotlib

```bash
pip install tensorflow numpy matplotlib
```

## Dataset
[Dataset name or source]. Expected structure:
