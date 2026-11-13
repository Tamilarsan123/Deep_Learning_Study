# Chapter 16.1 – Autoencoders: Theory and Scripting

Notes and hands-on code for autoencoders, from the underlying theory to Python implementations.

## Contents
- `<notebook>.ipynb`: implementation and experiments
- `<notes>`: theory notes

## Theory Covered
- What an autoencoder is: encoder → latent space (bottleneck) → decoder
- Objective: minimize reconstruction loss (MSE / binary cross-entropy)
- Undercomplete vs. overcomplete autoencoders
- Variants: Denoising, Sparse, Contractive, Variational (VAE)
- Applications: dimensionality reduction, denoising, anomaly detection, feature learning

## Implementation
- Framework: TensorFlow/Keras *(update if PyTorch)*
- Dataset: MNIST / Fashion-MNIST *(update)*
- Steps:
  1. Load and normalize data
  2. Build the encoder and decoder
  3. Compile with Adam and the reconstruction loss
  4. Train and plot the loss curves
  5. Visualize original vs. reconstructed images

## Requirements
`python>=3.9`, `numpy`, `matplotlib`, `tensorflow`

## Usage
    pip install -r requirements.txt
    jupyter notebook

## Results
Reconstruction loss and sample outputs go here *(add screenshots)*.

## Reference
Hands-On Machine Learning (Géron), Ch. 17 *(update if different)*
