import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="Diabetes Risk Prediction", layout="wide")

# ---------------------------
# Load Models & Scaler
# ---------------------------
rf_model = joblib.load("diabetes_model.joblib")
lr_model = joblib.load("logistic_model.joblib")
scaler = joblib.load("scaler.joblib")

# ---------------------------
# Title
# ---------------------------
st.title("🩺 Diabetes Risk Prediction System")

st.warning(
    "⚠️ This tool is for educational purposes only and should not be used "
    "as a medical diagnosis. Always consult a healthcare professional."
)

st.write(
    "Predict diabetes risk using Machine Learning models. "
    "Select a model and enter patient data in the sidebar."
)

# ---------------------------
# Sidebar Inputs
# ---------------------------
st.sidebar.header("Patient Information")

pregnancies = st.sidebar.number_input("Pregnancies", 0, 20, 1)
glucose = st.sidebar.number_input("Glucose Level", 0, 300, 120)
blood_pressure = st.sidebar.number_input("Blood Pressure", 0, 200, 70)
skin_thickness = st.sidebar.number_input("Skin Thickness", 0, 100, 20)
insulin = st.sidebar.number_input("Insulin Level", 0, 900, 80)
bmi = st.sidebar.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.sidebar.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
age = st.sidebar.number_input("Age", 1, 120, 30)

model_choice = st.sidebar.radio(
    "Select Model",
    ["Random Forest", "Logistic Regression"]
)

# ---------------------------
# Prepare Input
# ---------------------------
input_data = np.array([[
    pregnancies, glucose, blood_pressure,
    skin_thickness, insulin, bmi, dpf, age
]])

input_scaled = scaler.transform(input_data)

if model_choice == "Random Forest":
    model = rf_model
else:
    model = lr_model

prediction = model.predict(input_scaled)
probabilities = model.predict_proba(input_scaled)

# ---------------------------
# Display Prediction
# ---------------------------
st.subheader("📊 Prediction Result")

if prediction[0] == 1:
    st.error("⚠️ High Risk of Diabetes")
else:
    st.success("✅ Low Risk of Diabetes")

st.subheader("Confidence Scores")
st.write(f"No Diabetes: {probabilities[0][0]:.2f}")
st.write(f"Diabetes: {probabilities[0][1]:.2f}")

# ---------------------------
# ROC Curve
# ---------------------------
st.subheader("📈 ROC Curve")

# Reload dataset for ROC visualization
columns = [
    'Pregnancies', 'Glucose', 'BloodPressure',
    'SkinThickness', 'Insulin', 'BMI',
    'DiabetesPedigreeFunction', 'Age', 'Outcome'
]

data = pd.read_csv("Data/pima-indians-diabetes.csv", header=None, names=columns)

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

X_scaled = scaler.transform(X)

fpr, tpr, _ = roc_curve(y, model.predict_proba(X_scaled)[:, 1])
auc_score = roc_auc_score(y, model.predict_proba(X_scaled)[:, 1])

fig, ax = plt.subplots()
ax.plot(fpr, tpr)
ax.plot([0, 1], [0, 1], linestyle="--")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title(f"ROC Curve (AUC = {auc_score:.2f})")

st.pyplot(fig)

# ---------------------------
# Feature Importance (RF only)
# ---------------------------
if model_choice == "Random Forest":
    st.subheader("📌 Feature Importance")

    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": rf_model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    fig2, ax2 = plt.subplots()
    ax2.barh(importance_df["Feature"], importance_df["Importance"])
    ax2.invert_yaxis()
    ax2.set_title("Model Feature Importance")

    st.pyplot(fig2)