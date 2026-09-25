# Convolutional Neural Networks

This project focuses on building and training a CNN from scratch for gesture image classification, and comparing it with augmentation strategies and transfer learning.

## Installation

Install all required dependencies:

```bash
pip3 install -r requirements.txt
```

Run the notebook or Python scripts after installing the dependencies.

## What was done

* Explored the Zindi gesture image dataset: checked class balance and screened for duplicate images by file hash.
* Built a stratified train/validation split (67% / 33%).
* Implemented a custom PyTorch `Dataset` (`GestureDataset`) that reads images with OpenCV and exposes them through a `DataLoader`.
* Compared image resizing strategies (direct resize, crop + resize, padding + resize) and GPU-cached (`TensorDataset`) loading vs. on-the-fly loading.
* Rebuilt the **LeNet-5 architecture from scratch** (`my_LeNet`), with a configurable number of conv blocks, channel widths, dropout, and activation function.
* Used **Optuna** to tune the architecture and training hyperparameters (channels, dropout, linear layers, activation, learning rate, weight decay), with results persisted to SQLite.
* Trained the tuned baseline model and evaluated it with Cross-Entropy loss and **ROC AUC** (macro, one-vs-rest).
* Applied **Albumentations** augmentations (horizontal flip, affine transforms, brightness/contrast) during training and compared validation ROC AUC against the non-augmented baseline.
* Implemented **MixUp** and **CutMix** (and a random per-batch choice between the two) and measured their effect on validation score.
* Added **Test-Time-Augmentation** (averaging predictions on the original and horizontally flipped image) and measured the resulting ROC AUC gain.
* Fine-tuned pretrained **ResNet-18** and **ResNet-50** backbones (via `timm`) as a transfer-learning baseline, for comparison with the from-scratch model.

## Main techniques

`Convolutional Neural Networks` · `LeNet` · `Transfer Learning` · `ResNet` · `Albumentations` · `MixUp` · `CutMix` · `Test-Time-Augmentation` · `Optuna` · `ROC AUC` · `PyTorch` · `OpenCV`

## Dataset

* Zindi gesture dataset — hand gesture image classification (9 classes)