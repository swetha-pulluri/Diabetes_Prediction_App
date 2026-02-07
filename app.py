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
if st.button("🔍 Predict Diabetes Risk"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] * 100

    st.subheader("🧪 Prediction Result")

    if probability < 30:
        st.markdown(
            f"""
            <div style="background-color:#d4edda;padding:15px;border-radius:10px">
                <h3 style="color:#155724;">🟢 LOW RISK (SAFE)</h3>
                <p><b>Diabetes Probability:</b> {probability:.2f}%</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.progress(min(int(probability), 100))
        st.info("✅ **Health Tip:** Maintain a healthy diet and regular exercise.")

    elif 30 <= probability <= 60:
        st.markdown(
            f"""
            <div style="background-color:#fff3cd;padding:15px;border-radius:10px">
                <h3 style="color:#856404;">🟡 MODERATE RISK</h3>
                <p><b>Diabetes Probability:</b> {probability:.2f}%</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.progress(min(int(probability), 100))
        st.warning("⚠️ **Health Tip:** Monitor glucose levels and improve lifestyle habits.")

    else:
        st.markdown(
            f"""
            <div style="background-color:#f8d7da;padding:15px;border-radius:10px">
                <h3 style="color:#721c24;">🔴 HIGH RISK</h3>
                <p><b>Diabetes Probability:</b> {probability:.2f}%</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.progress(min(int(probability), 100))
        st.error("🚨 **Health Tip:** Please consult a healthcare professional.")

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



