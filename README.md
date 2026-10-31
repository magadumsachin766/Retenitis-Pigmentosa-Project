VGG-16 From Scratch for Eye Disease Classification
A PyTorch implementation of a custom VGG-16 architecture built from scratch for binary eye disease classification. The model is designed to classify retinal fundus images into:

Disease (Retinitis Pigmentosa)
Normal
The project includes:

Automatic dataset path detection for Kaggle and Colab
Data augmentation pipeline
Custom VGG-16 implementation
Class imbalance handling
Weighted sampling
Label smoothing
Early stopping
Learning rate scheduling
Confusion matrix visualization
Training history visualization
Project Overview
This project uses retinal fundus images to train a deep convolutional neural network based on the VGG-16 architecture.

Unlike transfer learning approaches, this implementation trains a VGG-style network entirely from scratch using PyTorch.

The system automatically:

Detects dataset directories.
Loads disease and normal samples.
Creates train/validation/test splits.
Performs image augmentation.
Trains a custom VGG-16 model.
Saves the best-performing checkpoint.
Evaluates the model on unseen test data.
Generates performance visualizations.
Model Architecture
The architecture follows the original VGG-16 design:

Feature Extractor
Block	Layers
Block 1	64 → 64 Conv + MaxPool
Block 2	128 → 128 Conv + MaxPool
Block 3	256 → 256 → 256 Conv + MaxPool
Block 4	512 → 512 → 512 Conv + MaxPool
Block 5	512 → 512 → 512 Conv + MaxPool
Classifier
512×7×7
    ↓
Linear(25088 → 2048)
    ↓
BatchNorm
    ↓
ReLU
    ↓
Dropout
    ↓
Linear(2048 → 2048)
    ↓
BatchNorm
    ↓
ReLU
    ↓
Dropout
    ↓
Linear(2048 → 2)
Dataset Structure
The code automatically searches for:

Disease Images
dataset_retinitis_pigmentosa_enhanced/
Normal Images
eye_diseases_classification/
└── eye_diseases_classification/
    └── normal/
Example:

datasets/

├── dataset_retinitis_pigmentosa_enhanced/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...

└── eye_diseases_classification/
    └── eye_diseases_classification/
        └── normal/
            ├── img1.jpg
            ├── img2.jpg
            └── ...
Features
Automatic Dataset Discovery
The script searches multiple locations:

SEARCH_ROOTS = [
    "/kaggle/input",
    "/content",
    ".",
    ".."
]
Compatible with:

Kaggle Notebooks
Google Colab
Local machines
Data Augmentation
Training images undergo:

Random crop
Horizontal flip
Vertical flip
Random rotation
Color jitter
Random affine transformation
Random grayscale conversion
Cutout regularization
Class Imbalance Handling
To reduce bias toward majority classes:

WeightedRandomSampler
WeightedRandomSampler(...)
Class-Weighted Loss
CrossEntropyLoss(weight=loss_w)
Regularization Techniques
Implemented methods:

Batch Normalization
Dropout
Label Smoothing
Cutout Augmentation
Weight Decay
Gradient Clipping
Early Stopping
Learning Rate Scheduling
ReduceLROnPlateau
Automatically reduces learning rate when validation loss stops improving.

Configuration
CONFIG = {
    "train_ratio": 0.80,
    "val_ratio": 0.10,
    "test_ratio": 0.10,
    "image_size": 224,
    "batch_size": 32,
    "num_epochs": 50,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "dropout_conv": 0.4,
    "dropout_fc": 0.5,
    "label_smoothing": 0.1,
    "grad_clip": 1.0,
    "early_stop_patience": 10,
    "num_workers": 2,
    "seed": 42
}
Installation
Clone Repository
git clone https://github.com/yourusername/vgg16-eye-disease-classification.git

cd vgg16-eye-disease-classification
Create Environment
conda create -n eye_vgg python=3.10
conda activate eye_vgg
or

python -m venv venv

source venv/bin/activate
Install Dependencies
pip install torch torchvision
pip install numpy matplotlib seaborn
pip install scikit-learn
pip install tqdm
Running the Project
Run:

python train.py
The script will:

Detect datasets
Create splits
Train VGG16
Save checkpoints
Evaluate model
Generate visualizations
Output Files
Best Model
best_vgg16_eye.pth
Contains:

{
    "epoch": ...,
    "model_state": ...,
    "val_loss": ...,
    "val_acc": ...
}
Training Curves
Generated file:

training_history.png
Shows:

Training Loss
Validation Loss
Training Accuracy
Validation Accuracy
Confusion Matrix
Generated file:

confusion_matrix.png
Example:

                Predicted
               D      N

Actual D      TP     FN
Actual N      FP     TN
Evaluation Metrics
The model reports:

Accuracy
Precision
Recall
F1-Score
Support
Using:

classification_report()
Example:

              precision    recall  f1-score   support

disease         0.96       0.95      0.95     200
normal          0.92       0.94      0.93     100

accuracy                             0.94     300
Hardware Requirements
Recommended:

Component	Requirement
GPU	NVIDIA T4 / P100 / RTX Series
VRAM	8 GB+
RAM	16 GB+
Storage	5 GB+
Works on CPU but training will be significantly slower.

Project Workflow
Dataset Discovery
        │
        ▼
Image Collection
        │
        ▼
Train/Val/Test Split
        │
        ▼
Data Augmentation
        │
        ▼
Weighted Sampling
        │
        ▼
VGG16 Training
        │
        ▼
Validation
        │
        ▼
Checkpoint Saving
        │
        ▼
Testing
        │
        ▼
Metrics & Visualizations
Future Improvements
Multi-class eye disease classification
Transfer Learning (ResNet, EfficientNet, DenseNet)
Attention Mechanisms
Vision Transformers (ViT)
Explainable AI using Grad-CAM
K-Fold Cross Validation
Mixed Precision Training (AMP)
ONNX/TensorRT deployment
Technologies Used
Python
PyTorch
Torchvision
NumPy
Scikit-learn
Matplotlib
Seaborn
TQDM
