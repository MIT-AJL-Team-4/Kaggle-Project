# ResNet50_ImageNet_SkinClassifier

## Overview
This deep learning model classifies **skin conditions** using **ResNet50** pretrained on **ImageNet**. It is designed to be **fair, accurate, and robust** across diverse skin tones as part of the **BTT AI Kaggle Competition**.

## Model Architecture
- **Base Model**: ResNet50 (pretrained on ImageNet, frozen during initial training)
- **Custom Layers**:
  - `GlobalAveragePooling2D()`
  - `Dense(1024, activation='relu')`
  - `Dropout(0.5)` (to reduce overfitting)
  - `Dense(num_classes, activation='softmax')`

## Training Details
- **Loss Function**: Sparse Categorical Crossentropy
- **Optimizer**: Adam
- **Metrics**: Accuracy, Weighted F1-score
- **Epochs**: 32 (with Early Stopping & Learning Rate Reduction)
- **Class Imbalance Handling**: `compute_class_weight()`

## Performance & Evaluation
- **Weighted F1-score** for fairness evaluation
- **Fairness Analysis** using `MetricFrame`
- **Grad-CAM Visualization** to interpret model decisions

