# Expert-Systems
##### For intelligent programming

# Data Exploration and Feature Analysis

This script performs exploratory data analysis (EDA) and feature importance visualization on a dataset, focusing on selected features from a prior feature selection process.

## Libraries

The script utilizes the following Python libraries:

* `pandas`: For data manipulation and analysis.
* `seaborn`: For statistical data visualization.
* `matplotlib.pyplot`: For creating plots and visualizations.

## Data Description

The script assumes that a pandas DataFrame `df` is already loaded and contains the dataset. Additionally, `selected_features` is assumed to be a list or array of column names that were determined to be important features through a previous feature selection step (e.g., using `SelectKBest` and `f_classif`). `selector` object is also assumed to be available from the feature selection step, so that the scores are available.

## Code Explanation

1.  **Descriptive Statistics:**

    ```python
    print(df.describe())
    ```

    * This line prints the descriptive statistics of the DataFrame `df`, including count, mean, standard deviation, minimum, quartiles, and maximum values for each numerical column. This provides a summary of the data's central tendency and dispersion.

2.  **Correlation Heatmap:**

    ```python
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.show()
    ```

    * This section generates a heatmap visualizing the correlation matrix of the DataFrame `df`.
    * `df.corr()` calculates the pairwise correlation of columns.
    * `sns.heatmap()` creates the heatmap, with annotations (`annot=True`), a color map (`cmap="coolwarm"`), and formatting (`fmt=".2f"`).
    * This heatmap helps identify highly correlated features.

3.  **Histograms and Boxplots:**

    ```python
    for column in selected_features:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.histplot(df[column], kde=True)
        plt.title(f"Histogram of {column}")
        
        plt.subplot(1, 2, 2)
        sns.boxplot(x=df[column])
        plt.title(f"Boxplot of {column}")
        plt.show()
    ```

    * This loop iterates through each feature in `selected_features` and generates two plots for each: a histogram and a boxplot.
    * `sns.histplot()` displays the distribution of the feature, with a kernel density estimate (`kde=True`).
    * `sns.boxplot()` shows the distribution of the feature and identifies potential outliers.

4.  **Feature Importance Plot:**

    ```python
    feature_scores = selector.scores_[selector.get_support()]
    sns.barplot(x=selected_features, y=feature_scores)
    plt.xticks(rotation=45)
    plt.title("Feature Importance Based on ANOVA F-value")
    plt.show()
    ```

    * This section creates a bar plot visualizing the importance of the selected features.
    * `selector.scores_[selector.get_support()]` retrieves the scores (ANOVA F-values) from the feature selection process for the selected features.
    * `sns.barplot()` generates the bar plot, with feature names on the x-axis and their scores on the y-axis.
    * `plt.xticks(rotation=45)` rotates the x-axis labels for better readability.
    * This plot helps understand the relative importance of the selected features based on the ANOVA F-value.  

### Output

![Image](https://github.com/user-attachments/assets/e12e4c22-dfb5-4967-99f3-2a9df5da5748)



# Exploratory Data Analysis and Feature Importance Visualization

This Python script performs exploratory data analysis (EDA) and visualizes feature importance for a given dataset (`df`), focusing on a set of pre-selected features (`selected_features`).

## Libraries

The script uses the following Python libraries:

* `pandas`: For data manipulation and analysis.
* `seaborn`: For statistical data visualization.
* `matplotlib.pyplot`: For creating plots and visualizations.

## Code Explanation

1.  **Descriptive Statistics:**

    ```python
    print(df.describe())
    ```

    * This line prints the descriptive statistics of the DataFrame `df`. It provides a summary of the numerical data, including count, mean, standard deviation, minimum, maximum, and quartiles. This helps understand the central tendency and spread of the data.

2.  **Correlation Heatmap:**

    ```python
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.show()
    ```

    * This section generates a heatmap visualizing the correlation matrix of the features in `df`.
    * `df.corr()` calculates the pairwise correlation coefficients between columns.
    * `sns.heatmap()` creates the heatmap, where the color intensity represents the correlation strength. `annot=True` adds the correlation values to the heatmap cells, `cmap="coolwarm"` sets the color scheme, and `fmt=".2f"` formats the annotations to two decimal places.
    * This plot helps identify highly correlated features, which can be useful for feature selection or understanding relationships between variables.

3.  **Histograms and Boxplots:**

    ```python
    for column in selected_features:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        sns.histplot(df[column], kde=True)
        plt.title(f"Histogram of {column}")
        
        plt.subplot(1, 2, 2)
        sns.boxplot(x=df[column])
        plt.title(f"Boxplot of {column}")
        plt.show()
    ```

    * This loop iterates through each feature in `selected_features` and generates a histogram and boxplot for each.
    * `sns.histplot()` creates a histogram showing the distribution of the feature's values, with a kernel density estimate (`kde=True`) overlaid.
    * `sns.boxplot()` creates a boxplot showing the distribution of the feature and highlighting potential outliers.
    * These plots help visualize the distribution and identify outliers for each selected feature.

4.  **Feature Importance Plot:**

    ```python
    feature_scores = selector.scores_[selector.get_support()]
    sns.barplot(x=selected_features, y=feature_scores)
    plt.xticks(rotation=45)
    plt.title("Feature Importance Based on ANOVA F-value")
    plt.show()
    ```

    * This section generates a bar plot visualizing the importance of the selected features based on their ANOVA F-values.
    * `selector.scores_[selector.get_support()]` retrieves the F-values from a feature selection process (e.g., `SelectKBest` with `f_classif`) for the selected features. `selector` object is assumed to be available from the feature selection step.
    * `sns.barplot()` creates a bar plot with the feature names on the x-axis and their corresponding F-values on the y-axis.
    * `plt.xticks(rotation=45)` rotates the x-axis labels for better readability.
    * This plot helps understand the relative importance of each selected feature in predicting the target variable.
## OUTPUT

![Image](https://github.com/user-attachments/assets/f5a6fb85-4314-4dff-b5b4-467868ae4aa6)

![image](https://github.com/user-attachments/assets/1cdac50b-9f8e-4432-b31b-60ea245c26a4)


![Image](https://github.com/user-attachments/assets/6c7d82d2-6abf-41d4-bde2-cccf4a63f828)


![Image](https://github.com/user-attachments/assets/1e0ac46f-4be9-4ad8-a824-9347e296317a)


![Image](https://github.com/user-attachments/assets/f06396dc-1cda-41ca-aa64-380eed12489a)



![Image](https://github.com/user-attachments/assets/59f698e5-f6ac-4124-afea-3e3b5f11bee8)



![Image](https://github.com/user-attachments/assets/110fef69-1a43-4078-8696-df36d14535e1)




![image](https://github.com/user-attachments/assets/23ddb4d3-427b-4954-b7be-6c6b8d5c85b1)



![Image](https://github.com/user-attachments/assets/178cbf73-c522-455a-a4d4-8198b94661a4)



# Reading and Displaying the First Few Rows of Cleaned Data

This script reads a CSV file named "cleaned_data.csv" into a pandas DataFrame and displays the first few rows to inspect the data.

## Libraries

The script uses the `pandas` library for data manipulation and analysis.

## Code Explanation

```python
import pandas as pd

df_cleaned = pd.read_csv("cleaned_data.csv")
df_cleaned.head()
```

![Image](https://github.com/user-attachments/assets/7c135334-c806-48f2-bfeb-0b251ad3d62b)



# Health Risk Assessment Expert System

This Python script implements a simple expert system for assessing health risks using the `experta` library. It takes user input about various health factors and predicts their risk level based on predefined rules.

## Libraries

The script utilizes the `experta` library for building the expert system.

## Code Description

The script consists of two main parts:

1.  **HealthRiskAssessment Class:**
    * This class defines the expert system's knowledge base and inference engine.
    * It inherits from `KnowledgeEngine` from the `experta` library.
    * It defines a series of rules using the `@Rule` decorator. Each rule specifies conditions (facts) and actions (declaring new facts).
    * The rules cover various health risk factors, including cholesterol levels, age, blood pressure, smoking, exercise, BMI, diet, alcohol consumption, stress, sleep, and family history.
    * The rules categorize risk levels as "low," "moderate," "high," or "very high."
    * The `print_risk` rule prints the predicted risk level to the console.

2.  **assess_risk Function:**
    * This function takes user input about their health factors.
    * It creates an instance of the `HealthRiskAssessment` class and resets the engine.
    * It prompts the user to enter values for cholesterol, age, blood pressure, smoking habits, exercise frequency, BMI, diet, alcohol consumption, stress levels, sleep duration, and family history of heart disease.
    * It declares facts based on the user's input.
    * It runs the inference engine to apply the rules and determine the risk level.

3.  **Main Execution Block:**
    * The `if __name__ == "__main__":` block ensures that the `assess_risk` function is called when the script is executed.

## Rules Breakdown

* **High Risk:**
    * Cholesterol > 240 and Age > 50
    * Blood Pressure > 140 and Smoking = "yes"
    * Diet = "unhealthy" and BMI > 30
    * Smoking = "yes" and Alcohol = "frequent"
    * Stress = "high" and Sleep < 5
    * Exercise = "none" and Diet = "unhealthy"
* **Very High Risk:**
    * Blood Pressure > 160 and Age > 60
* **Low Risk:**
    * Exercise = "regular" and BMI < 25
* **Moderate Risk:**
    * Family History = "yes"

## How to Run

1.  Ensure you have Python and the `experta` library installed (`pip install experta`).
2.  Save the script as a Python file (e.g., `health_risk.py`).
3.  Run the script from the command line: `python health_risk.py`.
4.  Follow the prompts to enter your health information.
5.  The script will output the predicted risk level.

## Example

![image](https://github.com/user-attachments/assets/ebd2c78a-646d-4f72-b7dd-6109fa5a993f)

# Print Python Executable Path

This script prints the absolute path of the Python interpreter executable.

## Code

```python
import sys
print(sys.executable)
```

![image](https://github.com/user-attachments/assets/70d0fa21-ea54-4b84-8da9-b2fdb655de26)

# Install the 'experta' Library

This command uses the Python interpreter to install the `experta` library using `pip`, the Python package installer.

## Code Explanation

```bash
!{sys.executable} -m pip install experta
```

![Image](https://github.com/user-attachments/assets/959573f0-eb8e-407d-9964-a710d6d31b87)


# Install the 'experta' Library

This command uses `pip`, the Python package installer, to install the `experta` library.

## Code Explanation

```bash
!pip install experta
```
![image](https://github.com/user-attachments/assets/19615b48-2fb9-4515-9c42-b9cdd4e72be5)

# Heart Disease Prediction using Decision Tree Classifier

This Python script performs heart disease prediction using a Decision Tree Classifier. It includes data loading, preprocessing, model training, evaluation, and saving the model.

## Libraries

The script utilizes the following Python libraries:

* `pandas`: For data manipulation and analysis.
* `sklearn.model_selection`: For splitting the data into training and testing sets, and for hyperparameter tuning.
* `sklearn.tree`: For the Decision Tree Classifier model.
* `sklearn.metrics`: For model evaluation metrics.
* `sklearn.preprocessing`: For scaling numerical features.
* `joblib`: For saving the trained model.
* `pickle`: For saving the trained model (alternative).
* `os`: for operating system dependent functionality.

## Code Description

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
import joblib
import pickle
import os

```
# Data Preparation for Heart Disease Prediction

This script prepares the heart disease dataset for machine learning model training. It involves loading the data, defining features and the target variable, scaling numerical features, and splitting the data into training and testing sets.

## Libraries

The script uses the `pandas` and `sklearn.model_selection` and `sklearn.preprocessing` libraries.

## Code Description

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Define features and target
df = pd.read_csv(r"C:\Users\Acer\Downloads\heart (1).csv")
df

X = df.drop(columns=['target'], axis=1)  # Features
y = df['target']  # Target (Diabetic or Not)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


```
# Train Decision Tree Classifier

This script trains a Decision Tree Classifier on the prepared training data.

## Libraries

The script uses the `sklearn.tree` library.

## Code Description

```python
from sklearn.tree import DecisionTreeClassifier

# Train Decision Tree Classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

dt_classifier = DecisionTreeClassifier(random_state=42)
```

# Hyperparameter Tuning for Decision Tree Classifier using GridSearchCV

This script defines a parameter grid for hyperparameter tuning of a Decision Tree Classifier using GridSearchCV.

## Libraries

The script uses no external libraries, as it only defines a dictionary.

## Code Description

```python
# Hyperparameter tuning using GridSearchCV
param_grid = {
    "max_depth": [3, 5, 7, 10],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

```
# Model Evaluation and Hyperparameter Tuning with GridSearchCV

This script evaluates the performance of a Decision Tree Classifier and performs hyperparameter tuning using GridSearchCV to improve its performance.

## Libraries

The script utilizes the following libraries:

* `sklearn.metrics`: For calculating the accuracy of the model.
* `sklearn.model_selection`: For performing hyperparameter tuning using GridSearchCV.
* `sklearn.tree`: For the Decision Tree Classifier model.

## Code Description

```python
# Evaluate the model
y_pred = clf.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

grid_search = GridSearchCV(dt_classifier, param_grid, cv=5, scoring="accuracy")
grid_search.fit(X_train, y_train)

# Get the best model
best_dt_classifier = grid_search.best_estimator_

# Train the model
best_dt_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = best_dt_classifier.predict(X_test)

```
![image](https://github.com/user-attachments/assets/aacad200-87cf-42ed-954f-587b099425a9)



# Model Evaluation Metrics

This script calculates and prints various evaluation metrics for a classification model's performance.

## Libraries

The script utilizes the following libraries:

* `sklearn.metrics`: For calculating accuracy, precision, recall, and F1-score.

## Code Description

```python
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
```

![image](https://github.com/user-attachments/assets/cc6c3252-7938-44b8-b2e6-d1db38f9b9da)




# Create Directory for Machine Learning Model

This script creates a directory named "ml_model" if it doesn't already exist.

## Libraries

The script uses the `os` library for operating system-dependent functionality.

## Code Description

```python
import os

os.makedirs("ml_model", exist_ok=True)

```

# Save Trained Decision Tree Classifier Model

This script saves a trained Decision Tree Classifier model to a file using `joblib`. It also includes error handling for potential issues during the saving process.

## Libraries

The script utilizes the `joblib` library for saving the trained model and the `os` library for interacting with the operating system.

## Code Description

```python
import joblib
import os

# Save the trained model
try:
    joblib.dump(best_dt_classifier, "ml_model/heart (1).pkl")
    print("Model saved successfully!")
except PermissionError:
    print("Error: Permission denied. Please check the file path and permissions.")
except Exception as e:
    print(f"An error occurred: {e}")
```
![image](https://github.com/user-attachments/assets/18d6b9f2-7514-4112-8e76-41bd5a7cd1d3)

# Install the 'streamlit' Library

This command uses `pip`, the Python package installer, to install the `streamlit` library.

## Code Explanation

```bash
!pip install streamlit

```


![image](https://github.com/user-attachments/assets/9109e693-f00b-4a08-bfd0-a7017ef1f04b)


# Heart Disease Prediction and Expert System with Streamlit

This Python script combines machine learning for heart disease prediction with an expert system for risk assessment, and creates an interactive web application using Streamlit.

## Libraries

The script utilizes the following libraries:

* `pandas`: For data manipulation and analysis.
* `numpy`: For numerical operations.
* `sklearn.model_selection`: For splitting the data into training and testing sets.
* `sklearn.tree`: For the Decision Tree Classifier model.
* `sklearn.metrics`: For model evaluation metrics.
* `experta`: For building the expert system.
* `joblib`: For loading the saved machine learning model.
* `pickle`: For loading the saved machine learning model.
* `streamlit`: For creating the web application.

## Code Description

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from experta import Fact, KnowledgeEngine, Rule, Field
from joblib import load
import pickle
import streamlit as st
```


# Data Preparation for Heart Disease Prediction

This script prepares the heart disease dataset for machine learning model training. It involves separating features and the target variable and splitting the data into training and testing sets.

## Libraries

The script utilizes the `pandas` and `sklearn.model_selection` libraries.

## Code Description

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Assuming df is already loaded from a CSV file
X = df.drop('target', axis=1)  # Features
y = df['target']  # Target variable (heart disease)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
import pandas as pd

df = pd.read_csv(r'C:\Users\Acer\Downloads\heart (1).csv')


dt_model = joblib.load("ml_model/heart (1).pkl")

# Evaluate the Decision Tree model
y_pred_dt = dt_model.predict(X_test)

dt_accuracy = accuracy_score(y_test, y_pred_dt)

dt_precision = precision_score(y_test, y_pred_dt)

dt_recall = recall_score(y_test, y_pred_dt)

dt_f1 = f1_score(y_test, y_pred_dt)


print("Decision Tree Model Performance:")

print(f"Accuracy: {dt_accuracy:.2f}")

print(f"Precision: {dt_precision:.2f}")

print(f"Recall: {dt_recall:.2f}")

print(f"F1-Score: {dt_f1:.2f}")



![image](https://github.com/user-attachments/assets/8d2ef919-d03b-4bb8-9119-f52b833273ef)


# Rule-Based Expert System Evaluation
from experta import KnowledgeEngine, Rule, Fact, P, Field

import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class HeartDiseaseFact(Fact):

    age = Field(int, mandatory=True)
    
    sex = Field(int, mandatory=True)
    cp = Field(int, mandatory=True)
    trestbps = Field(int, mandatory=True)
    chol = Field(int, mandatory=True)
    fbs = Field(int, mandatory=True)
    restecg = Field(int, mandatory=True)
    thalach = Field(int, mandatory=True)
    exang = Field(int, mandatory=True)
    oldpeak = Field(float, mandatory=True)
    slope = Field(int, mandatory=True)
    ca = Field(int, mandatory=True)
    thal = Field(int, mandatory=True)

class HeartDiseaseExpertSystem(KnowledgeEngine):
    @Rule(Fact(chol=lambda x: x > 240, age=lambda x: x > 50))
    def high_risk_rule1(self):
        self.risk = "high"

    @Rule(Fact(trestbps=lambda x: x > 140, exang=1))
    def high_risk_rule2(self):
        self.risk = "high"

    @Rule(Fact(thalach=lambda x: x < 100, oldpeak=lambda x: x > 2))
    def high_risk_rule3(self):
        self.risk = "high"

    @Rule(Fact(slope=2, cp=3))
    def high_risk_rule4(self):
        self.risk = "high"

    @Rule(Fact())
    def default_risk(self):
        self.risk = "medium"

# Evaluate the Expert System
expert_system = HeartDiseaseExpertSystem()

expert_system_predictions = []

for index, row in X_test.iterrows():

    expert_system.reset()
    
    expert_system.declare(Fact(
    
        age=row['age'],
        
        sex=row['sex'],
        
        cp=row['cp'],
        
        trestbps=row['trestbps'],
        
        chol=row['chol'],
        
        fbs=row['fbs'],
        
        restecg=row['restecg'],
        
        thalach=row['thalach'],
        
        exang=row['exang'],
        
        oldpeak=row['oldpeak'],

        slope=row['slope'],
        
        ca=row['ca'],
        
        thal=row['thal']
        
    ))
    expert_system.run()
    
    expert_system_predictions.append(1 if expert_system.risk == "high" else 0)


# Convert expert system predictions to binary (1 for high risk, 0 for low/medium)
expert_system_predictions = np.array(expert_system_predictions)

# Evaluate Expert System Performance
accuracy = accuracy_score(y_test, expert_system_predictions)

precision = precision_score(y_test, expert_system_predictions)

recall = recall_score(y_test, expert_system_predictions)

f1 = f1_score(y_test, expert_system_predictions)

print("Expert System Performance:")

print(f"Accuracy: {accuracy:.2f}")

print(f"Precision: {precision:.2f}")

print(f"Recall: {recall:.2f}")

print(f"F1-Score: {f1:.2f}")



# Evaluate the Expert System
es_accuracy = accuracy_score(y_test, expert_system_predictions)

es_precision = precision_score(y_test, expert_system_predictions)

es_recall = recall_score(y_test, expert_system_predictions)

es_f1 = f1_score(y_test, expert_system_predictions)

print("\nExpert System Performance:")

print(f"Accuracy: {es_accuracy:.2f}")

print(f"Precision: {es_precision:.2f}")

print(f"Recall: {es_recall:.2f}")

print(f"F1-Score: {es_f1:.2f}")



![image](https://github.com/user-attachments/assets/6be1a7d5-5c99-420a-aee5-f2aacbdca452)

# Compare the two systems

comparison_df = pd.DataFrame({

    'Model': ['Decision Tree', 'Expert System'],
    
    'Accuracy': [dt_accuracy, es_accuracy],
    
    'Precision': [dt_precision, es_precision],
    
    'Recall': [dt_recall, es_recall],
    
    'F1-Score': [dt_f1, es_f1]
})


print("\nModel Comparison:")

print(comparison_df)

![image](https://github.com/user-attachments/assets/b2d9b124-2c51-4abb-a39f-b97d711ef346)

