# Diabetic Retinopathy Severity Classifier

## Project Overview

This project uses machine learning to classify retinal fundus photographs according to the severity of diabetic retinopathy.

The model uses the APTOS 2019 Blindness Detection dataset and MobileNetV2 transfer learning.

## Dataset

Dataset: APTOS 2019 Blindness Detection

Source: Kaggle

Number of images: 3,662

The five classes are:

- No DR
- Mild
- Moderate
- Severe
- Proliferative DR

The dataset contains retinal fundus photographs with labels showing the severity of diabetic retinopathy.

## Model

The project uses MobileNetV2 with transfer learning.

The MobileNetV2 base model was pretrained on ImageNet and kept frozen during training.

The model has five output classes corresponding to the five diabetic retinopathy severity levels.

Input image size:

224 × 224 pixels

The model uses MobileNetV2 preprocessing before making predictions.

## Training

The dataset was divided into:

- 70% training
- 15% validation
- 15% testing

Training used:

- Optimizer: Adam
- Learning rate: 0.0001
- Batch size: 32
- Epochs: 10
- Loss: Sparse Categorical Crossentropy

Training was performed using TensorFlow 2.21.0.

## Model Results

The final validation accuracy was 71.77%.

On the test dataset, the model achieved:

- Accuracy: 73.64%
- Precision: 72.86%
- Recall: 73.64%
- F1-score: 68.45%

The model performed best on the No DR class.

The model had more difficulty distinguishing Mild, Severe and Proliferative DR from Moderate DR.

## Streamlit Application

The Streamlit application allows a user to:

1. Upload a retinal fundus image.
2. View the uploaded image.
3. Get a predicted diabetic retinopathy class.
4. See the model confidence.
5. Read a simple explanation of the prediction.

The application uses the same preprocessing as the trained model.

## Files

```text
Medical/
├── app.py
├── aptos_mobilenetv2.keras
├── requirements.txt
├── README.md
├── train.csv
└── train_images/