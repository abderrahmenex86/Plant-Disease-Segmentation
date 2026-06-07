# Segmenti

<div align="center">
  <p>
    <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch" />
    <img src="https://img.shields.io/badge/SMP-0.3.3-lightgrey?style=for-the-badge&logo=github&logoColor=white" alt="SMP" />
    <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  </p>
</div>

Segmenti is a crop disease leaf segmentation engine built with PyTorch. It generates the spatial segmentation masks utilized directly by the [FloraLens](https://github.com/abderrahmenex86/FloraLens) on-device visualization layout to delineate infected regions on plant foliage.

## Features

- **Double-Architecture Implementation**:
  - **U-Net**: Configured via `segmentation_models_pytorch` utilizing an ImageNet-pretrained MobileNetV2 encoder.
  - **DeepLabV3**: Built using `deeplabv3_mobilenet_v3_large` with custom auxiliary and classifier heads.
- **Delineation Objective**: Leverages a custom `MulticlassDiceLoss` function to maintain training stability when managing background-to-foreground class imbalances.
- **Synchronized Mask Processing**: Implements updated PyTorch `torchvision.transforms.v2` to apply color jitters, resizing ($520 \times 520$), and vertical/horizontal spatial flips concurrently to both the input leaf images and their target mask tensors.
- **Inference Parameterization**: Configures training using different learning rates (`1e-5` for the encoder/backbone features, `1e-3` for the decoder and segmentation heads) to preserve pretrained feature extractor representations.
- **Segmentation Metrology**: Evaluates boundary precision and overlap metrics via Dice Score and Mean Intersection-over-Union (Mean IoU), intentionally excluding background pixels.

## Tech Stack

- **Machine Learning Framework:** PyTorch, TorchVision
- **Library wrappers:** Segmentation Models PyTorch (SMP)
- **Metrics Evaluation:** TorchMetrics (Dice Score, Mean IoU)
- **Data Engineering:** TorchVision V2 Transforms, Pillow
- **Progress Tracking:** tqdm

## Getting Started

### Prerequisites
- Python (v3.10+)
- CUDA-compatible GPU (highly recommended for training)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/abderrahmenex86/segmenti.git
cd segmenti
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Prepare the Dataset:**
Place the `plantsegv3` images and matching target masks in the directory path below:
```text
dataset/plantsegv3/
├── train/
└── val/
```

4. **Run Training:**
```bash
python train.py
```
This script initializes the dataset loaders, starts optimization loops over 40 epochs, and exports the optimized weights output to `disease_segmentation_model.pth`.

## Related Projects

- [FloraLens](https://github.com/abderrahmenex86/FloraLens) — Offline plant, pest, and disease diagnosis app
- [Flora](https://github.com/abderrahmenex86/flora) — Plant classification model
- [Pesti](https://github.com/abderrahmenex86/pesti) — Pest classification model
