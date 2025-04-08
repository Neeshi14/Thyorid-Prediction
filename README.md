# Thyroid Cancer Prediction Model

This repository contains two machine learning models for predicting thyroid cancer diagnosis (Benign or Malignant) using the "thyroid_cancer_risk_data.csv" dataset from Kaggle. The models implemented are Logistic Regression and a Neural Network built with TensorFlow Keras.

## Overview

This project aims to develop and compare the performance of Logistic Regression and a Neural Network model in predicting thyroid cancer diagnosis based on patient data.

## Dataset

The dataset used is the "thyroid_cancer_risk_data.csv" dataset, available on Kaggle.

## Features

The dataset includes the following features:

* **Patient_ID:** Unique identifier for each patient. (Dropped during training)
* **Age:** Patient’s age in years.
* **Gender:** Patient’s gender (Male/Female). (Label Encoded to 0/1)
* **Country:** Patient’s country of residence. (Dropped during training)
* **Ethnicity:** Patient’s ethnic background. (Dropped during training)
* **Family_History:** Whether the patient has a family history of thyroid cancer (Yes/No). (Converted to 0/1)
* **Radiation_Exposure:** Whether the patient has been exposed to radiation (Yes/No). (Converted to 0/1)
* **Iodine_Deficiency:** Whether the patient has iodine deficiency (Yes/No). (Converted to 0/1)
* **Smoking:** Whether the patient is a smoker (Yes/No). (Converted to 0/1)
* **Obesity:** Whether the patient is classified as obese (Yes/No). (Converted to 0/1)
* **Diabetes:** Whether the patient has diabetes (Yes/No). (Converted to 0/1)
* **TSH_Level:** Level of thyroid-stimulating hormone (TSH) in blood (mIU/L).
* **T3_Level:** Level of triiodothyronine (T3) hormone in blood (nmol/L).
* **T4_Level:** Level of thyroxine (T4) hormone in blood (nmol/L).
* **Nodule_Size:** Size of thyroid nodules (cm).
* **Thyroid_Cancer_Risk:** Assessed risk level (Low, Medium, High). (Ordinal Encoded to 0, 1, 2)
* **Diagnosis:** Final thyroid diagnosis (Benign, Malignant). (Target variable, converted to 0/1)

## Dependencies

* Python 3.x
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* TensorFlow
* Keras

## Installation
1.  **Download the dataset:**

    Download the dataset from Kaggle.These is the link of the dataset [here](https://www.kaggle.com/datasets/mzohaibzeeshan/thyroid-cancer-risk-dataset).

## Usage

1.  **Run the model:**
   
    * Load and preprocess the dataset.
    * Convert categorical features to numerical.
    * Split the dataset into training and testing sets.
    * Train and evaluate the Logistic Regression model.
    * Train and evaluate the Neural Network model.
    * Display the accuracy, classification reports, confusion matrix, and sample results.

## Model Architectures

### Logistic Regression

* **Type:** Logistic Regression
* **Features:** 13 features (after preprocessing)
* **Loss Function:** Logistic loss

### Neural Network (TensorFlow Keras)

* **Type:** Sequential Neural Network
* **Input Layer:** 13 input features
* **Hidden Layers:** Three dense layers with 5, 30, and 50 neurons, respectively, using ReLU activation.
* **Output Layer:** One neuron with sigmoid activation for binary classification.
* **Loss Function:** Binary cross-entropy
* **Optimizer:** Adam

## Training Details

### Logistic Regression

* **Train/Test Split:** 80/20

### Neural Network

* **Epochs:** 2
* **Batch Size:** 6
* **Train/Test Split:** 80/20

## Evaluation

The models' performances are evaluated on the test set, and the following metrics are reported:

* **Accuracy**
* **Classification Report** (Precision, recall, F1-score)
* **Confusion Matrix**

### Results

* **Logistic Regression:**
    * Testing Accuracy: 82.56%
    * Training Accuracy: 82.78%
* **Neural Network:**
    * Testing Accuracy: 78.97%


