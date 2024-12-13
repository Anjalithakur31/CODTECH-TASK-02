# CODTECH-TASK-02

**Name**: Anjali Thakur Rakeshbhai  
**Company**: CODETECH IT Solutions  
**ID**: CT08DS421  
**Domain**: Artificial Intelligence  
**Duration**: December 5th 2024 - January 5th 2025  
**Mentor**: Muzammil Ahmed

## Overview

This Python script performs machine learning tasks using preprocessed employee management data (`processed_employee_management.csv`). The script builds and evaluates three classification models: Logistic Regression, Decision Tree, and Random Forest. The models are trained on the dataset and evaluated on various performance metrics like accuracy, precision, recall, and F1 score. This task demonstrates how different models can be used to classify data and evaluate their performance in a consistent manner.

## Key Features of the Script

1. **Data Loading**:  
   Loads preprocessed data from a CSV file (`processed_employee_management.csv`) containing feature and target variables.

2. **Data Splitting**:  
   Splits the dataset into feature variables (X) and the target variable (y). The data is then divided into training and testing sets with an 80-20 split.

3. **Model Initialization**:  
   Three machine learning models are initialized:  
   - **Logistic Regression**  
   - **Decision Tree Classifier**  
   - **Random Forest Classifier**

4. **Model Training & Prediction**:  
   Each model is trained on the training dataset and used to make predictions on the test dataset.

5. **Model Evaluation**:  
   The performance of each model is evaluated using the following metrics:
   - Accuracy  
   - Precision  
   - Recall  
   - F1 Score  
   - Classification Report (including more detailed metrics for each class)

6. **Model Comparison**:  
   The script prints the evaluation results for each model, allowing for a comparison of their performances.

## Technologies Used

1. **Pandas**:  
   A Python library for data manipulation, used here to load and split the dataset.

2. **Scikit-learn**:  
   A machine learning library that provides tools for:
   - `train_test_split`: Splitting the data into training and testing sets.
   - `LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`: Different machine learning models.
   - `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `classification_report`: Evaluation metrics to assess the models.

3. **CSV**:  
   The dataset is stored in CSV format for easy data exchange and processing.

4. **OUTPUT**:
   ![image](https://github.com/user-attachments/assets/59d6c377-2b42-4833-ba76-fa831b20783e)
   ![image](https://github.com/user-attachments/assets/e8511a4d-0b6b-4ffe-84c8-0d019aabea10)
