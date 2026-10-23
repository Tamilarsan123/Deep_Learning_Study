# Chapter 14.3 – AlexNet (2012) Architecture

Study notes and implementation of **AlexNet**, the CNN that won ImageNet LSVRC-2012 (top-5 error **15.3%**) and started the deep learning revolution in computer vision.

Part of my [Deep_Learning_Study](https://github.com/Tamilarsan123/Deep_Learning_Study) series.

---

## What this covers

- AlexNet architecture, layer by layer
- Key innovations and why they mattered
- Implementation of the network and shape tracing through each layer

## Architecture

Input: `227 × 227 × 3` RGB image (the paper states 224; 227 gives correct dimensions)

| # | Layer | Config | Output |
|---|-------|--------|--------|
| 1 | Conv | 96 filters, 11×11, stride 4, ReLU | 55×55×96 |
|   | Max Pool | 3×3, stride 2 | 27×27×96 |
| 2 | Conv | 256 filters, 5×5, pad 2, ReLU | 27×27×256 |
|   | Max Pool | 3×3, stride 2 | 13×13×256 |
| 3 | Conv | 384 filters, 3×3, pad 1, ReLU | 13×13×384 |
| 4 | Conv | 384 filters, 3×3, pad 1, ReLU | 13×13×384 |
| 5 | Conv | 256 filters, 3×3, pad 1, ReLU | 13×13×256 |
|   | Max Pool | 3×3, stride 2 | 6×6×256 |
| 6 | FC | 4096, ReLU, Dropout 0.5 | 4096 |
| 7 | FC | 4096, ReLU, Dropout 0.5 | 4096 |
| 8 | FC | 1000, Softmax | 1000 |

**5 convolutional + 3 fully connected layers, ~60 million parameters.**

## Key innovations

- **ReLU activation** – much faster training than tanh/sigmoid
- **Dropout (0.5)** – reduces overfitting in the FC layers
- **Overlapping max pooling** – 3×3 window with stride 2
- **Local Response Normalization (LRN)** – after the first two conv layers
- **Data augmentation** – random crops, horizontal flips, PCA colour jitter
- **GPU training** – original model was split across two GTX 580 GPUs

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
cd "Deep_Learning_Study/09 Chapter_14.3 Alexnet(2012) Architecture"
python <script_name>.py
```

> Replace `<script_name>` with the script in this folder.

## Key takeaways

- Depth + ReLU + GPUs + big data made deep CNNs practical.
- Large early filters (11×11, 5×5) later gave way to stacked 3×3 filters in VGG.
- AlexNet is the baseline for comparing VGG, GoogLeNet, and ResNet.

## Reference

Krizhevsky, A., Sutskever, I., Hinton, G. E. *ImageNet Classification with Deep Convolutional Neural Networks.* NeurIPS 2012.

## Author

**Tamilarasan K** – [GitHub](https://github.com/Tamilarasan123) · [LinkedIn](https://linkedin.com/in/tamilarasan-k-03a27834b)
