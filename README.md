# Expert-Systems
For intelligent programming


# Heart Disease Analysis Project

This project analyzes heart disease data using machine learning and expert system approaches.

## Project Description

The project involves the following key steps:

1.  **Data Preprocessing**: Cleaning, transforming, and preparing the heart disease dataset.
   
2.  **Feature Selection**: Identifying the most relevant features for predicting heart disease.
   
3.  **Model Development**: Training a Decision Tree Classifier to predict heart disease risk.
   
4.  **Expert System Implementation**: Developing a rule-based expert system to assess heart disease risk.
   
5.  **Model Evaluation**: Comparing the performance of the Decision Tree Classifier and the expert system.

## Data Preprocessing

The data preprocessing steps include:

* **Loading the Dataset**: Reading the data from a CSV file.
   
* **Handling Missing Values**: Filling missing values with the median.
   
* **Scaling Numerical Features**: Normalizing numerical features using MinMaxScaler.
   
* **Encoding Categorical Features**: Converting categorical features into numerical format using OneHotEncoder.
   
* **Feature Selection**: Selecting the top 10 features using SelectKBest and f\_classif.
   
* **Saving Cleaned Data**: Storing the processed data.

### Code: Data Preprocessing

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif

# Step 1: Load Dataset
file_path = r"C:\Users\Acer\Downloads\heart (1).csv"  # Correct file path
df = pd.read_csv(file_path)

df.fillna(df.median(), inplace=True)

numerical_features = df.select_dtypes(include=[np.number]).columns
df[numerical_features] = MinMaxScaler().fit_transform(df[numerical_features])

categorical_features = df.select_dtypes(include=["object"]).columns
encoder = OneHotEncoder(sparse_output=False, drop='first')
categorical_encoded = encoder.fit_transform(df[categorical_features])
categorical_df = pd.DataFrame(categorical_encoded, columns=encoder.get_feature_names_out(categorical_features))
df = df.drop(columns=categorical_features)
df = pd.concat([df, categorical_df], axis=1)

X = df.drop(columns=["target"])
y = df["target"]

selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X, y)

selected_features = X.columns[selector.get_support()]
cleaned_df = pd.DataFrame(X_selected, columns=selected_features)
cleaned_df["target"] = y
cleaned_df.to_csv("cleaned_data.csv", index=False)

print("Data Preprocessing Completed! Cleaned data saved as cleaned_data.csv")

![Image](https://github.com/user-attachments/assets/62c533b6-4745-46c7-a14b-79df49935293)
