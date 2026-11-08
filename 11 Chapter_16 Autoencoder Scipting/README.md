# Chapter 16 – Autoencoders (Scripting)

Hands-on implementations of autoencoders from my Deep Learning study series. Each script builds, trains, and evaluates an autoencoder variant, with reconstruction results.

## What is an Autoencoder?

An autoencoder is an unsupervised neural network that learns a compact representation of data. It has two parts:

- **Encoder** – compresses the input into a low-dimensional latent code (bottleneck).
- **Decoder** – reconstructs the input from that code.

Training minimizes reconstruction loss (e.g., MSE / binary cross-entropy) between input and output.

## Topics Covered

- Basic (undercomplete) autoencoder
- Stacked / deep autoencoder
- Denoising autoencoder
- Sparse autoencoder
- Convolutional autoencoder
- Variational autoencoder (VAE)

> Keep only the variants that exist in this folder.

## Project Structure

```
11 Chapter_16 Autoencoder Scipting/
├── <script_1>.py / .ipynb
├── <script_2>.py / .ipynb
└── README.md
```

## Requirements

- Python 3.9+
- TensorFlow / Keras (or PyTorch)
- NumPy
- Matplotlib

```bash
pip install tensorflow numpy matplotlib
```

## Usage

```bash
git clone https://github.com/Tamilarsan123/Deep_Learning_Study.git
cd "Deep_Learning_Study/11 Chapter_16 Autoencoder Scipting"
python <script_name>.py
```

## Dataset

MNIST / Fashion-MNIST (loaded via Keras; no manual download needed).

## Results

- Reconstruction loss: _add value_
- Sample input vs. reconstructed images: _add screenshot_

## Key Learnings

- Bottleneck size controls compression vs. reconstruction quality.
- Denoising forces the model to learn robust features.
- VAEs learn a probabilistic latent space that supports generation.

## Author

**Tamilarasan K**
GitHub: [Tamilarasan123](https://github.com/Tamilarasan123) · LinkedIn: [tamilarasan-k](https://linkedin.com/in/tamilarasan-k-03a27834b)
