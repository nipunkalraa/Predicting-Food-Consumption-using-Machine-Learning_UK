# Predicting-Food-Consumption-using-Machine-Learning_UK
This project uses machine learning techniques to predict whether individuals are likely to consume above-average amounts of red meat based on socio-demographic characteristics. The analysis helps identify key demographic factors that can inform targeted marketing strategies.

## Please check the 'Code_Predicting_Food_Consumption_using_ML.R' file for full reproducible code. The code has an extensive markdown to show what each function does.
![Screenshot (16)](https://github.com/user-attachments/assets/6941c4c9-ee3d-4952-bb53-452267535d83)


## 📊 Dataset
Individual-level survey data containing:

Demographic variables (Age, Sex, Ethnicity, Health status, Employment)

Food consumption patterns (including red meat, alcohol, food and vegetables, fish consumption)


## 📦 Packages Used
tidyverse: Data transformation and visualisation

caret: Machine learning algorithms and model evaluation

colorspace: Colour palette manipulation for visualisations

## 🔧 Methods
## Data Preprocessing:
Removed extreme age groups (0-15 and 75+)

Encoded categorical variables (One-Hot for Sex/Work, Ordinal for Age/Health)

Created binary outcome variable for red meat consumption (above/below average)

## Machine Learning Models:
Logistic Regression: Baseline binary classification

Random Forest: A more complex ensemble learning approach

Used 80/20 train/test split with 10-fold cross-validation

## Model Evaluation:
Accuracy, Kappa, Sensitivity, Specificity

Precision, Recall, F1 Score

Variable importance assessment

## 🔍 Key Findings
The most important predictors for red meat consumption:
Sex (Male)

Ethnicity (White)

Employment status (employed)

Age groups (45-64, 65-74)

The Random Forest model outperformed Logistic Regression across performance metrics
