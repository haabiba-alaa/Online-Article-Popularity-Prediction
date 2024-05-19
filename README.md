# Online Article Popularity Prediction

This repository contains our project on predicting the popularity of online articles using machine learning techniques. We tackled the problem using both regression and classification approaches to cover different aspects of popularity prediction. This README provides an overview of the project, including data preprocessing, feature selection, model implementation, and evaluation.

## Table of Contents

- [Overview](#overview)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Feature Selection](#feature-selection)
  - [Regression](#regression)
  - [Classification](#classification)
- [Modeling](#modeling)
  - [Regression Models](#regression-models)
  - [Classification Models](#classification-models)
- [Results](#results)
- [Conclusion](#conclusion)

## Overview

The aim of this project is to predict the popularity of online articles. We approached this task using both regression and classification methods. For regression, the target variable is the numerical popularity score, while for classification, we categorized articles into different popularity levels.

## Data Preprocessing

1. **Handling Null Values**: We removed any null values from the dataset.
2. **Duplicate Check**: We checked for and removed duplicate entries to ensure data quality.
3. **Encoding**: Categorical features were encoded to numerical values for model compatibility.

## Exploratory Data Analysis (EDA)

We conducted an extensive EDA to understand the distribution and relationships of features within the dataset. This included visualizations and statistical analyses to uncover patterns and insights that could inform our modeling approach.

## Feature Selection

Feature selection was a critical step in our process, ensuring that we used the most relevant features for our models.

### Regression

We employed five different techniques for feature selection in the regression task:

1. **Information Gain**: Determined the importance of numerical features based on the entropy of the target variable.
2. **RandomForestRegressor**: Utilized feature importance provided by a Random Forest model to rank numerical features.
3. **Gradient Boosting**: Similar to the Random Forest Regressor, Gradient Boosting was used to rank numerical features based on their importance.
4. **Correlation Matrix**: Assessed the linear relationship between numerical features and the target variable.
5. **ANOVA (Analysis of Variance)**: Applied to categorical features to determine their significance.

### Classification

For the classification task, we used the following feature selection techniques:

1. **Information Gain Classifier**: Determined the importance of numerical features based on the entropy of the target variable.
2. **RandomForestClassifier**: Utilized feature importance provided by a Random Forest model to rank numerical features.
3. **Gradient Boosting Classifier**: Used to rank numerical features based on their importance.
4. **Chi-Square**: Assessed the association between categorical variables, determining if there is a significant relationship by comparing observed and expected frequencies.

## Modeling

### Regression Models

We implemented the following regression models:

- **Linear Regression**
- **Lasso Regression**
- **XGBoost**
- **Random Forest**

### Classification Models

For classification, we used the following models:

- **Logistic Regression**
- **Random Forest**
- **Gradient Boosting**
- **K-Nearest Neighbors (KNN)**

## Results

Detailed results for both regression and classification models are included in the results folder. Metrics such as R-squared, Mean Absolute Error (MAE), and Mean Squared Error (MSE) for regression, and accuracy, precision, recall, and F1-score for classification, were used to evaluate the performance of our models.

## Conclusion

Our models successfully predicted the popularity of online articles with varying degrees of accuracy. The feature selection techniques significantly improved model performance by focusing on the most relevant features.

## Acknowledgments
Thanks to the dedicated development team:

Habiba Alaa

Malak Raafat

Rana Wahid

Mariam Yossri

