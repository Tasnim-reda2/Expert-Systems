import streamlit as st
import pickle
import pandas as pd


# Load the trained model
data = pd.read_csv(r"C:/Users/Acer/Downloads/heart (1).csv")


# Streamlit UI
st.title("Heart Disease Risk Prediction")

# Input fields for user data
st.sidebar.header("Input Health Data")
age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=50)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
cp = st.sidebar.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
trestbps = st.sidebar.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
chol = st.sidebar.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", ["True", "False"])
restecg = st.sidebar.selectbox("Resting Electrocardiographic Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
thalach = st.sidebar.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=220, value=150)
exang = st.sidebar.selectbox("Exercise-Induced Angina", ["Yes", "No"])
oldpeak = st.sidebar.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=6.2, value=1.0)
slope = st.sidebar.selectbox("Slope of the Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])
ca = st.sidebar.number_input("Number of Major Vessels Colored by Fluoroscopy", min_value=0, max_value=3, value=0)
thal = st.sidebar.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])

# Convert categorical inputs to numerical values
sex = 1 if sex == "Male" else 0
cp_mapping = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
cp = cp_mapping[cp]
fbs = 1 if fbs == "True" else 0
restecg_mapping = {"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}
restecg = restecg_mapping[restecg]
exang = 1 if exang == "Yes" else 0
slope_mapping = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
slope = slope_mapping[slope]
thal_mapping = {"Normal": 3, "Fixed Defect": 6, "Reversible Defect": 7}
thal = thal_mapping[thal]

# Create a DataFrame from user input
user_data = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'cp': [cp],
    'trestbps': [trestbps],
    'chol': [chol],
    'fbs': [fbs],
    'restecg': [restecg],
    'thalach': [thalach],
    'exang': [exang],
    'oldpeak': [oldpeak],
    'slope': [slope],
    'ca': [ca],
    'thal': [thal]
})

# Predict using the model
if st.sidebar.button("Predict"):
    prediction = load_model.predict(user_data)
    if prediction[0] == 1:
        st.error("High Risk of Heart Disease")
    else:
        st.success("Low Risk of Heart Disease")

# Display user input data
st.write("### User Input Data")
st.write(user_data)

# Visualization (optional)
st.write("### Data Visualization")
st.write("Cholesterol vs Resting Blood Pressure")
st.scatter_chart(user_data, x='chol', y='trestbps')