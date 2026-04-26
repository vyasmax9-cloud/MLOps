import streamlit as st
import joblib
import pandas as pd
from huggingface_hub import hf_hub_download

# =========================
# LOAD MODEL FROM HF
# =========================
model_path = hf_hub_download(
    repo_id="vyasmax9/tourism-prediction",
    filename="tourism_model_v1.joblib"   # ✅ FIX NAME (check exact file name in HF)
)

model = joblib.load(model_path)

# =========================
# UI
# =========================
st.title("🌍 Tourism Package Prediction App")
st.write("Predict whether a customer will purchase the Wellness Tourism Package")

# =========================
# USER INPUTS
# =========================
age = st.number_input("Age", 18, 70, 30)
income = st.number_input("Monthly Income", 1000, 200000, 50000)

typeofcontact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
occupation = st.selectbox("Occupation", ["Salaried", "Freelancer"])
gender = st.selectbox("Gender", ["Male", "Female"])
citytier = st.selectbox("City Tier", [1, 2, 3])
maritalstatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
preferredpropertystar = st.selectbox("Preferred Property Star", [3, 4, 5])
designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager"])
productpitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "Luxury"])

children = st.number_input("Number of Children Visiting", 0, 5, 0)

# =========================
# PREDICTION
# =========================
if st.button("Predict"):

    # ✅ Create DataFrame (IMPORTANT)
    input_df = pd.DataFrame([{
        'Age': age,
        'NumberOfChildrenVisiting': children,
        'MonthlyIncome': income,
        'TypeofContact': typeofcontact,
        'Occupation': occupation,
        'Gender': gender,
        'CityTier': citytier,
        'MaritalStatus': maritalstatus,
        'PreferredPropertyStar': preferredpropertystar,
        'Designation': designation,
        'ProductPitched': productpitched
    }])

    # =========================
    # MODEL PREDICTION
    # =========================
    prediction = model.predict(input_df)[0]

    # If classifier
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(input_df)[0][1]
    else:
        prob = None

    # =========================
    # OUTPUT
    # =========================
    if prediction == 1:
        st.success("✅ Likely to Purchase")
    else:
        st.error("❌ Not Likely to Purchase")

    if prob is not None:
        st.write(f"Prediction Probability: {prob:.2f}")
