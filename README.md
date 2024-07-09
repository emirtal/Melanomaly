# Melanoma Detection Using Large Vision Models (LVMs)

This repository contains the code and resources for melanoma detection using advanced machine learning models, specifically CLIP and ResNet architectures. The project includes preprocessing steps, model training, evaluation, and a user-friendly dashboard for visualizing predictions and Grad-CAM heatmaps.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Preprocessing](#data-preprocessing)
  - [Training the Models](#training-the-models)
  - [Evaluating the Models](#evaluating-the-models)
  - [Dashboard](#dashboard)
- [Results](#results)
- [License](#license)

## Introduction

This project explores the use of Large Vision Models (LVMs) for melanoma detection. We leverage the CLIP model and ResNet-101 architecture, fine-tuned on dermatoscopic images, to classify images as benign or malignant. The repository also includes a dashboard for visualizing model predictions and Grad-CAM heatmaps for interpretability.

## Dataset

We use two datasets from Kaggle:
1. [Melanoma Balanced Dataset](https://www.kaggle.com/datasets/scipygaurav/melanoma-balanced-dataset)
2. [Skin Lesion Analysis Towards Melanoma Detection](https://www.kaggle.com/datasets/wanderdust/skin-lesion-analysis-toward-melanoma-detection)

Download these datasets and place them in a directory named `data/` within the root of the project.

## Requirements

- Python 3.12
- PyTorch 2.3.0+cu121
- CUDA 12.5
- Other Python dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/melanoma-detection.git
    cd melanoma-detection
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Ensure you have the necessary CUDA and cuDNN versions installed.

## Usage

### Data Preprocessing

Preprocess the dataset using the provided script:
```bash
python preprocess.py
```

5. Train the models
- Train CLIP: python train_clip.py
