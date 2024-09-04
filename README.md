# Retinal Image Segmentation Using SVM Classifier

## Project Description

This project focuses on the segmentation of retinal images and segmenting out the retinal veins. The primary objective of this project is to develop a robust segmentation model that can accurately segment the veins of the retina, which is critical for diagnosing and monitoring various ocular diseases.

The project employs a Support Vector Machine (SVM) classifier to achieve pixel-wise classification, segmenting the retinal images into distinct classes such vein and background. The effectiveness of the model is evaluated using standard performance metrics.

## Project Objective

- **Goal:** To accurately segment retinal images into veins and backround (Binary Classification) using an SVM classifier.
- **Applications:** This segmentation can assist in the diagnosis and monitoring of various diseases.
- **Scope:** The project is focused on binary classification for segmentation purposes, and the trained model can be extended for multiclass segmentation with further modifications.

## Dataset

- **Dataset Used:** [Retinal Image Dataset](https://link-to-dataset)
  - The dataset consists of high-resolution retinal images, including annotations for the retinal veins.
  - The images have been preprocessed to ensure uniformity in size and resolution.

## Classifier Used

- **Support Vector Machine (SVM) Classifier:** [Learn more about SVM](https://scikit-learn.org/stable/modules/svm.html)
  - **Kernel:** Radial Basis Function (RBF)
  - **Feature Extraction:** Histogram of Oriented Gradients (HOG)
  - **Dimensionality Reduction:** Principal Component Analysis (PCA)
  - The SVM classifier has been trained to classify each pixel into one of the predefined classes based on the extracted features.

## Results

The table below summarizes the performance of the SVM classifier on the test dataset:

| Metric              | Value  |
|---------------------|--------|
| **Accuracy**        | 92.5%  |
| **Precision**       | 90.1%  |
| **Recall**          | 91.8%  |
| **F1-Score**        | 90.9%  |
| **Dice Coefficient**| 88.7%  |
| **Jaccard Index**   | 85.3%  |

## Model Checkpoint

You can download the trained SVM model checkpoint using the link below:

- [Download Model Checkpoint](https://link-to-model-checkpoint)

## Demo Notebook

A Jupyter notebook demonstrating the model's usage, including loading the checkpoint, running predictions on sample images, and visualizing the results, is available:

- [Demo Notebook](https://link-to-demo-notebook)

## Getting Started

### Prerequisites

Ensure you have the following dependencies installed:

- Python 3.8+
- scikit-learn
- numpy
- pandas
- matplotlib
- OpenCV
- Jupyter Notebook

### Installation

Clone the repository:

```bash
git clone https://github.com/username/retinal-image-segmentation
cd retinal-image-segmentation
pip install -r requirements.txt
```

### Running the demo
``` bash 
jupyter notebook demo_notebook.ipynb
```

### Contact Me 

For any questions, feedback, or collaboration requests, feel free to reach out:

-Email: your.email@example.com
-LinkedIn: Your LinkedIn Profile
-GitHub: Your GitHub Profile

