import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
with open(r"C:\Users\Acer\Downloads\heart_model.pkl", "rb") as file:
    model = pickle.load(file)


st.set_page_config(page_title="Heart Disease Risk Predictor", layout="centered")

st.title("🫀 Heart Disease Risk Assessment")
st.markdown("""
This app predicts the **likelihood of heart disease** based on your medical information.  
Please fill in the following details to get your risk level.
""")

# Sidebar for user inputs
st.sidebar.header("User Input Features")


def user_input_features():
    age = st.sidebar.slider("Age", 20, 100, 40)
    sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
    cp = st.sidebar.selectbox("Chest Pain Type", [0, 1, 2, 3])
    trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    chol = st.sidebar.slider("Serum Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    restecg = st.sidebar.selectbox("Resting ECG Results", [0, 1, 2])
    thalach = st.sidebar.slider("Max Heart Rate Achieved", 60, 220, 150)
    exang = st.sidebar.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.sidebar.slider("ST Depression", 0.0, 6.0, 1.0)
    slope = st.sidebar.selectbox("Slope of ST Segment", [0, 1, 2])
    ca = st.sidebar.selectbox("Number of Major Vessels (0–3)", [0, 1, 2, 3])
    thal = st.sidebar.selectbox("Thalassemia", [0, 1, 2, 3])

    sex = 1 if sex == "Male" else 0

    data = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }

    features = pd.DataFrame(data, index=[0])
    return features


input_df = user_input_features()

st.subheader("Your Input Data")
st.write(input_df)

# Prediction
if st.button("Predict Risk"):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

    st.markdown("### Prediction Probability")
    st.write(f"Low Risk: {prediction_proba[0][0] * 100:.2f}%")
    st.write(f"High Risk: {prediction_proba[0][1] * 100:.2f}%")

    # Pie chart
    fig1, ax1 = plt.subplots()
    ax1.pie(prediction_proba[0], labels=["Low Risk", "High Risk"], autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    st.pyplot(fig1)

# Optional Dashboard: Simulated data distribution
st.markdown("### Example Feature Distribution")
if st.checkbox("Show Simulated Feature Distribution Charts"):
    df = pd.DataFrame({
        "age": np.random.normal(50, 10, 100),
        "chol": np.random.normal(250, 30, 100),
        "thalach": np.random.normal(150, 20, 100)
    })

    fig2, ax = plt.subplots(1, 3, figsize=(15, 4))
    sns.histplot(df["age"], bins=20, ax=ax[0], color="skyblue")
    ax[0].set_title("Age Distribution")

    sns.histplot(df["chol"], bins=20, ax=ax[1], color="salmon")
    ax[1].set_title("Cholesterol")

    sns.histplot(df["thalach"], bins=20, ax=ax[2], color="lightgreen")
    ax[2].set_title("Max Heart Rate")

    st.pyplot(fig2)
