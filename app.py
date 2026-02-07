import streamlit as st
import numpy as np
import pandas as pd
import pickle

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="wide"
)

# -----------------------------
# Load model
# -----------------------------
model = pickle.load(open("diabetes_model.pkl", "rb"))

# -----------------------------
# Title & description
# -----------------------------
st.markdown(
    """
    <div style="background-color:#f15a29;padding:15px;border-radius:10px">
        <h2 style="color:white;text-align:center;">
            🧠 ML Project by Swetha — XGBoost-based Diabetes Predictor
        </h2>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("## Diabetes Prediction App")
st.write(
    "This app uses a **Machine Learning model (XGBoost)** to predict whether "
    "a person is likely to have diabetes based on medical information."
)

# -----------------------------
# Sidebar inputs
# -----------------------------
st.sidebar.header("📝 Enter Patient Details")

pregnancies = st.sidebar.number_input(
    "Pregnancies (Normal: 0–6)", min_value=0, max_value=20, value=0
)
glucose = st.sidebar.number_input(
    "Glucose Level (Normal: 70–99 mg/dL)", min_value=0, max_value=300, value=0
)
blood_pressure = st.sidebar.number_input(
    "Blood Pressure (Normal: 70–80 mmHg)", min_value=0, max_value=200, value=0
)
skin_thickness = st.sidebar.number_input(
    "Skin Thickness (Normal: 10–30 mm)", min_value=0, max_value=100, value=0
)
insulin = st.sidebar.number_input(
    "Insulin Level (Normal: 16–166 µU/ml)", min_value=0, max_value=1000, value=0
)
bmi = st.sidebar.number_input(
    "BMI (Normal: 18.5–24.9)", min_value=0.0, max_value=70.0, value=0.0
)
dpf = st.sidebar.number_input(
    "Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.0
)
age = st.sidebar.number_input(
    "Age", min_value=1, max_value=120, value=21
)

# -----------------------------
# Prepare input data
# -----------------------------
input_data = np.array([[
    pregnancies,
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi,
    dpf,
    age
]])

# -----------------------------
# Prediction button
# -----------------------------
if st.button("🔍 Predict Diabetes Risk"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] * 100

    if prediction == 1:
        st.error(
            f"🔴 **High Risk of Diabetes**\n\n"
            f"📊 Probability: **{probability:.2f}%**"
        )
    else:
        st.success(
            f"🟢 **Low Risk of Diabetes**\n\n"
            f"📊 Probability: **{probability:.2f}%**"
        )

# -----------------------------
# Feature importance section
# -----------------------------
st.markdown("---")
st.subheader("📊 Feature Importance")

feature_names = [
    "Pregnancies",
    "Glucose",
    "Blood Pressure",
    "Skin Thickness",
    "Insulin",
    "BMI",
    "Diabetes Pedigree Function",
    "Age"
]

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

st.bar_chart(importance_df.set_index("Feature"))

# -----------------------------
# Disclaimer
# -----------------------------
st.markdown("---")
st.caption(
    "⚠️ **Disclaimer:** This application is for educational purposes only and "
    "should not be considered as medical advice. "
    "Please consult a qualified healthcare professional for diagnosis."
)
