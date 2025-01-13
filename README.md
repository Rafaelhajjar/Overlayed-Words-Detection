# Overlayed Words Detection

A deep learning system for detecting and classifying overlapping handwritten words using a dual-headed Convolutional Neural Network (CNN) architecture.

## Table of Contents
- [Overview](#overview)
- [Data Generation](#data-generation)
- [Model Architecture](#model-architecture)
- [Training Methodology](#training-methodology)
- [Testing and Evaluation](#testing-and-evaluation)
- [Installation and Usage](#installation-and-usage)
- [Mathematical Details](#mathematical-details)

## Overview

This project implements a system to detect and classify two overlapping handwritten words in images. The system utilizes a ResNet18 backbone with dual classification heads to simultaneously predict both words.

## Data Generation

### Synthetic Data Creation Process
1. **Image Generation Parameters**:
   - Image Size: 256×256 pixels
   - RGBA color space with transparency
   - Font: System default (Arial/Helvetica)
   - Font size: 40px

2. **Text Rendering Process**:
   - Words are rendered with semi-transparent colors:
     - First word: Red (RGB: 200,0,0, Alpha: 64)
     - Second word: Blue (RGB: 0,0,200, Alpha: 64)
   - Text positioning: Centered using bounding box calculations
   ```python
   x_pos = (IMAGE_SIZE[0] - text_width) // 2
   y_pos = (IMAGE_SIZE[1] - text_height) // 2
   ```

3. **Image Composition**:
   - Base transparent layer
   - Word 1 overlay with alpha blending
   - Word 2 overlay with alpha blending
   - Final conversion to RGB color space

## Model Architecture

### Backbone Network
- **Base Model**: ResNet18 (pretrained on ImageNet)
- **Feature Extraction**: 512-dimensional feature vector

### Dual Classification Heads
Each classification head consists of:
1. Linear layer (512 → 256 neurons)
2. Batch Normalization
3. ReLU activation
4. Dropout (p=0.5)
5. Output layer (256 → num_classes)

```
Feature Vector (512) → FC(256) → BatchNorm → ReLU → Dropout → FC(num_classes)
```

## Training Methodology

### Hyperparameters
- Batch Size: 32
- Learning Rate: 0.001
- Epochs: 20 (configurable)
- Optimizer: Adam
- Loss Function: Cross-Entropy

### Training Process
1. **Data Preprocessing**:
   - Image resizing to 256×256
   - Normalization via ToTensor transform
   - One-hot encoding of word labels

2. **Loss Calculation**:
   ```python
   total_loss = criterion(word1_pred, word1_labels) + criterion(word2_pred, word2_labels)
   ```

3. **Gradient Control**:
   - Gradient clipping with max norm = 1.0
   ```python
   clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

### Mathematical Details

#### Cross-Entropy Loss
For each word prediction:
```
L = -∑(y_i * log(p_i))

where:
- y_i is the true label (one-hot encoded)
- p_i is the predicted probability
- i ranges over all classes
```

#### Softmax Activation
Applied to final layer outputs:
```
σ(z)_i = exp(z_i) / ∑(exp(z_j))

where:
- z_i is the logit for class i
- j ranges over all classes
```

## Testing and Evaluation

### Metrics
1. **Per-Word Accuracy**:
   ```
   Accuracy_word = (Correct_predictions / Total_samples) * 100
   ```

2. **Evaluation Process**:
   - Forward pass through model
   - Argmax on softmax outputs
   - Comparison with ground truth labels
   - Separate accuracy tracking for each word position

## Installation and Usage

### Prerequisites
- Python 3.10 + 
- PyTorch
- torchvision
- PIL
- numpy
- pandas

### Running the Pipeline

1. **Generate Synthetic Data**:
```bash
python Image_generator --num_images 1000 --output_dir ./dataset --dict_file words.txt
```

2. **Train the Model**:
```bash
python train_model --data_dir ./dataset --dict_file words.txt
```

3. **Test the Model**:
```bash
python test_model --model_path ./dataset/trained_model.pth --test_data_dir ./dataset/test --dict_file words.txt
```
