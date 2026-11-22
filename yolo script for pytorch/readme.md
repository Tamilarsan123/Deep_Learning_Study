# YOLO Script for PyTorch

A PyTorch implementation for [training / inference / both] with YOLO for object detection, built as part of my deep learning study.

## Features
- YOLO model implementation in PyTorch
- [Custom dataset loading / data augmentation]
- [Training loop with loss and metrics]
- [Inference on images / video / webcam]

## Project Structure
```
yolo script for pytorch/
├── [script_name].py     # Main script
├── requirements.txt     # Dependencies
└── README.md
```

## Requirements
- Python 3.8+
- PyTorch
- torchvision
- OpenCV
- NumPy
- Matplotlib

```bash
pip install torch torchvision opencv-python numpy matplotlib
```

## Usage

**Clone the repo**
```bash
git clone https://github.com/Tamilarsan123/Deep_Learning_Study.git
cd "Deep_Learning_Study/yolo script for pytorch"
```

**Train**
```bash
python [script_name].py --data [path/to/dataset] --epochs 50 --batch-size 16
```

**Inference**
```bash
python [script_name].py --weights [weights.pt] --source [image_or_video]
```

## Dataset
[Dataset name, source link, and annotation format, e.g. YOLO txt / COCO / Pascal VOC.]

## Results
[Add sample output images, mAP, or loss curves here.]

## Learning Notes
- How YOLO divides an image into grids and predicts boxes and classes
- Loss functions: objectness, classification, bounding-box regression
- Non-max suppression (NMS) for filtering detections

## Author
**Tamilarasan K**
[GitHub](https://github.com/Tamilarasan123) · [LinkedIn](https://linkedin.com/in/tamilarasan-k-03a27834b)
