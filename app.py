import streamlit as st
import numpy as np
import pickle

# Load model
model = pickle.load(open("diabetes_model.pkl","rb"))

# Page configuration
st.set_page_config(page_title="Diabetes Prediction App", page_icon="🩺", layout="centered")
# Add a top banner
st.markdown("""
    <div style="background-color:#FF4B4B;padding:15px;border-radius:10px">
        <h2 style="color:white;text-align:center;">🧠 ML Project by Swetha — XGBoost-based Diabetes Predictor</h2>
    </div>
    """, unsafe_allow_html=True)


# Title and description
st.title("Diabetes Prediction App")
st.markdown("""
This app uses a Machine Learning model (XGBoost) to predict whether a person is likely to have **diabetes** 
based on their medical information.
""")

# Sidebar inputs with helpful medical ranges 🩺
st.sidebar.header("Enter Patient Details:")

preg = st.sidebar.number_input("Pregnancies (Normal: 0–6)", 0, 20)
glucose = st.sidebar.number_input("Glucose Level (Normal: 70–99 mg/dL)", 0, 200)
bp = st.sidebar.number_input("Blood Pressure (Normal: 70–80 mmHg)", 0, 150)
skin = st.sidebar.number_input("Skin Thickness (Normal: 10–30 mm)", 0, 100)
insulin = st.sidebar.number_input("Insulin Level (Normal: 16–166 mu U/ml)", 0, 900)
bmi = st.sidebar.number_input("BMI (Normal: 18.5–24.9)", 0.0, 70.0)
dpf = st.sidebar.number_input("Diabetes Pedigree Function (Normal: <0.5)", 0.0, 3.0)
age = st.sidebar.number_input("Age (Risk ↑ after 45 years)", 1, 120)

# Add a small note at the bottom of the sidebar
st.sidebar.markdown("""
---
**ℹ️ Note:**  
Values outside the normal range may indicate higher diabetes risk.
""")
# 🩺 Health Alerts Based on Input Values
st.sidebar.markdown("### ⚠️ Health Alerts")

# Glucose alert
if glucose > 126:
    st.sidebar.error("🔴 High Glucose Level detected (Possible Diabetes risk).")
elif glucose < 70:
    st.sidebar.warning("🟡 Low Glucose Level detected (Hypoglycemia risk).")
else:
    st.sidebar.success("🟢 Glucose Level is within normal range.")

# Blood Pressure alert
if bp > 90:
    st.sidebar.error("🔴 High Blood Pressure (Hypertension risk).")
elif bp < 60:
    st.sidebar.warning("🟡 Low Blood Pressure detected.")
else:
    st.sidebar.success("🟢 Blood Pressure is normal.")

# BMI alert
if bmi > 24.9:
    st.sidebar.error("🔴 High BMI (Overweight risk).")
elif bmi < 18.5:
    st.sidebar.warning("🟡 Low BMI (Underweight).")
else:
    st.sidebar.success("🟢 BMI is within healthy range.")

# Predict button
if st.sidebar.button("Predict"):
    data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    result = model.predict(data)[0]
    prob = model.predict_proba(data)[0]

    st.subheader("Prediction Result:")

    # Show confidence level
    if result == 1:
        st.error(f"🚨 The person is likely to have Diabetes.\n\n**Confidence:** {prob[1]*100:.2f}%")
    else:
        st.success(f"✅ The person is NOT likely to have Diabetes.\n\n**Confidence:** {(prob[0])*100:.2f}%")

    # Optional bar chart for probabilities
    st.subheader("Prediction Probability")
    st.bar_chart(prob)

