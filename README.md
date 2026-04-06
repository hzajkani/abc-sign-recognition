# ABC Sign Language Recognition

A deep learning project that classifies **German Sign Language (DGS)** hand gestures for letters **A**, **B**, and **C** using transfer learning with **EfficientNetV2S**, achieving **99.70% test accuracy**.

## Table of Contents

- [Project Overview](#project-overview)
- [Key Results](#key-results)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Notebooks](#notebooks)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Key Design Decisions](#key-design-decisions)
- [Technologies](#technologies)
- [License](#license)

## Project Overview

This project demonstrates a complete computer vision pipeline - from raw image data to a production-ready classifier. It was developed as a case study for the WBS Coding School Data Science program.

The core challenge: build a model that can recognize hand signs for letters A, B, and C from photos of **24 different individuals**, and generalize well to **people it has never seen before**.

### Approach Progression

| Step | Approach | Accuracy |
| --- | --- | --- |
| Baseline | Simple CNN from scratch | ~33-98% (unstable) |
| + Augmentation | CNN with data augmentation | ~62% |
| + Transfer Learning | EfficientNetB0 (frozen backbone) | ~99% |
| **Final Model** | **EfficientNetV2S (fine-tuned)** | **99.70%** |

## Key Results

- **Test Accuracy:** 99.70%
- **Backbone:** EfficientNetV2S (pre-trained on ImageNet)
- **Training Strategy:** Two-phase (head warmup + progressive fine-tuning)
- **Data Strategy:** Group-aware splitting to prevent data leakage

## Dataset

The dataset consists of **~884 hand gesture images** from **24 individuals** (students and instructors), each performing signs for letters A, B, and C.

**Important characteristics:**
- Images vary in resolution, lighting, and background
- Each person contributes multiple photos per letter
- Data is organized by person (group) to enable group-aware splitting

> **Note:** The dataset is not included in this repository due to its size (~2.7 GB). See the [Usage](#usage) section for data setup instructions.

### Data Structure

```
data/
  +-- images/
       +-- grouped/           # Raw images organized by person
       |    +-- Person_A/
       |    |    +-- A/       # Sign language letter A
       |    |    +-- B/       # Sign language letter B
       |    |    +-- C/       # Sign language letter C
       |    +-- Person_B/
       |    +-- ...
       +-- raw/               # After group-aware splitting
       |    +-- train/
       |    +-- val/
       |    +-- test/
       +-- processed/         # TensorFlow Dataset objects
            +-- train/
            +-- val/
            +-- test/
```

## Project Structure

```
abc-sign-recognition/
  +-- notebooks/
  |    +-- 01_data_preparation.ipynb      # Data splitting & baseline CNN
  |    +-- 02_image_augmentation.ipynb    # Augmentation techniques
  |    +-- 03_transfer_learning.ipynb     # Transfer learning tutorial
  |    +-- 04_final_model.ipynb           # Competition model (99.70%)
  +-- data/                               # (gitignored) Dataset & models
  +-- .gitignore
  +-- requirements.txt
  +-- LICENSE
  +-- README.md
```

## Notebooks

### [01 - Data Preparation](notebooks/01_data_preparation.ipynb)
- Understanding image labeling via directory structure
- **Group-aware data splitting** using Monte Carlo optimization
- Processing images into TensorFlow Dataset objects
- Training a baseline CNN from scratch

### [02 - Image Augmentation](notebooks/02_image_augmentation.ipynb)
- Data augmentation techniques (RandomFlip, RandomRotation, RandomContrast)
- Building a modular CNN pipeline (augmentor + preprocessor + extractor + classifier)
- Evaluating augmentation's impact with confusion matrices

### [03 - Transfer Learning](notebooks/03_transfer_learning.ipynb)
- Introduction to transfer learning with EfficientNetB0
- Freezing backbone weights and training a custom classification head
- Achieving ~99% accuracy with minimal training
- Fine-tuning the backbone for further improvement

### [04 - Final Competition Model](notebooks/04_final_model.ipynb)
- **EfficientNetV2S** backbone with progressive fine-tuning
- Smart augmentation (no horizontal flip - it changes sign meaning!)
- Two-phase training with intelligent callbacks
- Comprehensive evaluation: confusion matrix, classification report, misclassified image analysis
- **Final accuracy: 99.70%**

## Model Architecture

```
Input (256x256x3)
       |
Data Augmentation (Rotation, Contrast, Zoom, Brightness)
       |
EfficientNetV2S Preprocessor
       |
EfficientNetV2S Backbone (pre-trained on ImageNet, ~20.8M params)
       |
GlobalAveragePooling2D
       |
Dense(256, ReLU) -> BatchNorm -> Dropout(0.5)
       |
Dense(128, ReLU) -> BatchNorm -> Dropout(0.3)
       |
Dense(3, Softmax) -> Output: [A, B, C]
```

### Training Strategy

| Phase | What's Trained | Learning Rate | Purpose |
| --- | --- | --- | --- |
| Phase 1 | Classification head only | 1e-3 | Warm up the new layers |
| Phase 2 | Head + top 30% of backbone | 1e-5 | Adapt features to sign language |
| Phase 3 (optional) | Entire model | 5e-6 | Fine-tune all features |

## Installation

### Prerequisites

- Python 3.9+
- pip

### Setup

```bash
# Clone the repository
git clone https://github.com/hzajkani/abc-sign-recognition.git
cd abc-sign-recognition

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Prepare the Data

Place your grouped sign language images in the following structure:

```
data/images/grouped/
  +-- Person_Name/
       +-- A/    # Images of sign A
       +-- B/    # Images of sign B
       +-- C/    # Images of sign C
```

### 2. Run the Notebooks in Order

1. **`01_data_preparation.ipynb`** - Splits data and creates TensorFlow datasets
2. **`02_image_augmentation.ipynb`** - (Optional) Explore augmentation effects
3. **`03_transfer_learning.ipynb`** - (Optional) Learn transfer learning basics
4. **`04_final_model.ipynb`** - Train the final competition model

### 3. Use the Trained Model

```python
import tensorflow as tf

model = tf.keras.models.load_model('data/models/kamran_v3_best.keras')
# Preprocess your image to 256x256 and predict
predictions = model.predict(image_batch)
```

## Key Design Decisions

### Group-Aware Data Splitting
Standard random splitting would leak information - the same person's hand could appear in both training and test sets. Our Monte Carlo-based group-aware split ensures **entire groups (people)** stay within a single split, providing realistic evaluation of generalization to new individuals.

### No Horizontal Flip in Augmentation
Unlike general image classification, **flipping a sign language gesture horizontally can change its meaning**. The final model deliberately excludes horizontal flipping from augmentation to preserve sign semantics.

### EfficientNetV2S as Backbone
Chosen for the optimal balance of accuracy and model size:
- **EfficientNetB0**: Fast but slightly lower accuracy
- **EfficientNetV2S**: Best accuracy-to-size ratio (99.70%, ~238MB)
- **InceptionResNetV2**: 100% test accuracy but likely overfitting, 3x larger

### Two-Phase Training
Training the classification head first (with frozen backbone) prevents the randomly initialized head from corrupting the pre-trained backbone weights through backpropagation. Once the head is warmed up, we carefully fine-tune the backbone with a much lower learning rate.

## Technologies

- **Python 3.11**
- **TensorFlow / Keras** - Deep learning framework
- **EfficientNetV2S** - Pre-trained backbone (ImageNet)
- **NumPy** - Numerical computing
- **Matplotlib / Seaborn** - Visualization
- **scikit-learn** - Evaluation metrics
- **Jupyter Notebook** - Interactive development
