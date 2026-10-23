# Chapter 14.2 – LeNet-5 Architecture

Study notes and implementation of **LeNet-5**, the pioneering CNN by Yann LeCun et al. (1998) for handwritten digit recognition. It introduced the conv → pool → fully connected pattern that modern CNNs still follow.

Part of my [Deep_Learning_Study](https://github.com/Tamilarsan123/Deep_Learning_Study) series.

---

## What this covers

- LeNet-5 architecture, layer by layer
- Why convolution + pooling works for images
- Implementation of the network and training on MNIST

## Architecture

Input: `32 × 32 × 1` grayscale image (MNIST digits padded from 28×28)

| Layer | Type | Config | Output |
|-------|------|--------|--------|
| C1 | Conv | 6 filters, 5×5, stride 1, tanh | 28×28×6 |
| S2 | Avg Pool | 2×2, stride 2 | 14×14×6 |
| C3 | Conv | 16 filters, 5×5, stride 1, tanh | 10×10×16 |
| S4 | Avg Pool | 2×2, stride 2 | 5×5×16 |
| C5 | Conv | 120 filters, 5×5, tanh | 1×1×120 |
| F6 | FC | 84 units, tanh | 84 |
| Output | FC | 10 units, Softmax | 10 |

**~61,000 trainable parameters** (very small compared to modern networks).

## Key ideas

- **Local receptive fields** – each neuron looks at a small 5×5 patch
- **Weight sharing** – the same filter slides across the image, cutting parameters
- **Subsampling (pooling)** – reduces spatial size and adds small shift invariance
- **Hierarchical features** – edges → strokes → digit shapes

## Original vs modern implementation

| Aspect | Original (1998) | Modern |
|--------|-----------------|--------|
| Activation | Scaled tanh | tanh or ReLU |
| Pooling | Trainable average pooling | Average or max pooling |
| C3 connectivity | Partial (not all maps connected) | Full |
| Output layer | RBF units | Softmax |

## Requirements

```
python >= 3.8
tensorflow / keras   # or pytorch, depending on the script
numpy
matplotlib
```

## Usage

```bash
git clone https://github.com/Tamilarsan123/Deep_Learning_Study.git
cd "Deep_Learning_Study/08 Chapter_14.2 LeNet 5_Architecture"
python <script_name>.py
```

> Replace `<script_name>` with the script in this folder.

## Key takeaways

- LeNet-5 proved CNNs work for real-world recognition (check and digit reading).
- Its design is the blueprint for AlexNet, VGG, and later architectures.
- Limited by 1990s compute and data, which AlexNet (2012) overcame.

## Reference

LeCun, Y., Bottou, L., Bengio, Y., Haffner, P. *Gradient-Based Learning Applied to Document Recognition.* Proceedings of the IEEE, 1998.

## Author

**Tamilarasan K** – [GitHub](https://github.com/Tamilarasan123) · [LinkedIn](https://linkedin.com/in/tamilarasan-k-03a27834b)
