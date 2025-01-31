# Melanoma Detection Using Large Vision Models (LVMs)

This repository contains the code and resources for melanoma detection using advanced machine learning models, specifically CLIP and ResNet architectures. The project includes preprocessing steps, model training, evaluation, and a user-friendly dashboard for visualizing predictions and Grad-CAM heatmaps.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
  - [Training the Models](#training-the-models)
  - [Optional: Use Trained Models](#optional-use-trained-models)
  - [Dashboard](#dashboard)
- [Training Your Own Model](#training-your-own-model)

## Introduction

This project explores the use of Large Vision Models (LVMs) for melanoma detection. We leverage the CLIP model and ResNet-101 architecture, fine-tuned on dermatoscopic images, to classify images as benign or malignant. The repository also includes a dashboard for visualizing model predictions and Grad-CAM heatmaps for interpretability.

## Dataset

We use two datasets from Kaggle:
1. [Melanoma Balanced Dataset](https://www.kaggle.com/datasets/scipygaurav/melanoma-balanced-dataset)
2. [Skin Lesion Analysis Towards Melanoma Detection](https://www.kaggle.com/datasets/wanderdust/skin-lesion-analysis-toward-melanoma-detection)

The two datasets need to be modified so that they fall into two classes: 
- Benign
- Malignant

Once complete, move the data in the appropriate folders under the Image_data directory. Make sure not to rename or move the directories unless the paths are modified in the training scripts.

## Requirements

- Python 3.12
- PyTorch 2.3.0+cu121
- CUDA 12.5
- Other Python dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
    ```bash
    git clone [https://github.com/yourusername/melanoma-detection.git](https://github.com/emirtal/Melanomaly.git)
    cd melanoma-detection
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
OR use IDE and env of choice. 

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

4. Ensure you have the necessary CUDA and Pytorch versions installed. If specific versions are unavailable for your system, alternative Pytorch versions should work. If your system does not contain an Nvidia GPU capable of CUDA, the scripts will automatically use the CPU in place. If this is the case, you should consider reducing the size of the dataset. 

## Usage

### Training the models
Note: Place the data in the "Image_data" directory. Make sure to modify classes and hyperparameters if you would like to train other types of data. 

Train CLIP:
```bash
python clip_Train.py
```

Train ResNet-101:
```bash
python ResNet_Train.py
```

### Optional: Use Trained Models
If you do not wish to retrain the models, you can use the same trained models created during our study. Due to file size limitations, we have segmented the large files into smaller parts. Follow the instructions below to reconstruct the original file.

### For Unix-like Systems

1. **Reconstructing the File**

   Navigate to the directory containing the file segments and run:

   ```bash
   cat segment_* > best_clip_classifier.pth
   cat segment_* > best_resnet_classifier.pth

### For Windows: 

1. **Reconstruting the File**
   
   ```bash
   Open PowerShell and run the following script, replacing the paths as necessary:
   $destinationFile = "C:\path\to\large_file_reconstructed.tar"
   $sourceFolder = "C:\path\to\segments"
   $segmentFiles = [System.IO.Directory]::GetFiles($sourceFolder, "segment_*")
   $outputStream = [System.IO.File]::OpenWrite($destinationFile)
   foreach ($segmentFile in $segmentFiles) {
    $buffer = [System.IO.File]::ReadAllBytes($segmentFile)
    $outputStream.Write($buffer, 0, $buffer.Length)
   }
   $outputStream.Close() ```

***Place reconstructed files under /Melanomaly/Models***

### Dashboard
To run the dashboard locally:
```bash
streamlit run Melanomaly_Dashboard.py
```

### Training Your Own Model

If you wish to train your own model with different data, it is advised you maintain the directory structure provided in this repo. Please keep in mind, our structure is using 2 classes of data and if you wished to add more classes, you will need to modify "num_classes = " found in the train and prediction scripts. 
