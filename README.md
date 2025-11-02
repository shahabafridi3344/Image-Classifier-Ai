# AI Project 1: Image Classifier using CNN

## Overview
This project implements an image classifier using PyTorch. 
It trains a Convolutional Neural Network (CNN) on the CIFAR-10 dataset to classify images into 10 categories.

## Features
- Built using PyTorch and TorchVision
- Supports GPU acceleration (CUDA)
- Displays accuracy and loss curves
- Easy to extend for other datasets

## Installation
Run the following command in Jupyter or terminal to install dependencies:
```python
import sys
!{sys.executable} -m pip install torch torchvision tqdm matplotlib
```

## Usage
Simply run the provided notebook cells to:
1. Download CIFAR-10 dataset
2. Define the CNN model
3. Train the model and view accuracy

## Example Output
```
Epoch 1: Train Loss = 1.56, Validation Accuracy = 45%
Epoch 5: Validation Accuracy = 73%
```

## Files
- `image_classifier.ipynb`: Main Jupyter notebook code
- `README_Image_Classifier.txt`: Project documentation
