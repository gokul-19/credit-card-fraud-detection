import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from io import BytesIO

# -------------------------------
# STREAMLIT CONFIG
# -------------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection")

# -------------------------------
# LOAD PIPELINE MODEL FROM GITHUB
# -------------------------------
MODEL_URL = "https://raw.githubusercontent.com/SUHAASSHETTY/Fraud_Detection_Using_AI/main/fraud_detection_pipeline.pkl"

@st.cache_data
def load_model():
    try:
        resp = requests.get(MODEL_URL)
        resp.raise_for_status()
        model = joblib.load(BytesIO(resp.content))
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_model()
if model:
    st.success("✅ Model loaded successfully!")
else:
    st.warning("⚠️ Model not loaded. Check the URL or internet connection.")

# -------------------------------
# TRANSACTION INPUT FORM
# -------------------------------
st.header("🔍 Enter Transaction Details")

col1, col2, col3 = st.columns(3)

with col1:
    trans_type = st.selectbox("Transaction Type", ["PAYMENT","TRANSFER","CASH_OUT","CASH_IN","DEBIT","OTHER"])
    amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0)

with col2:
    old_org = st.number_input("Sender Old Balance ($)", min_value=0.0, value=1000.0)
    new_org = st.number_input("Sender New Balance ($)", min_value=0.0, value=900.0)

with col3:
    old_dest = st.number_input("Receiver Old Balance ($)", min_value=0.0, value=500.0)
    new_dest = st.number_input("Receiver New Balance ($)", min_value=0.0, value=600.0)

# Compute balance differences
diff_org = old_org - new_org
diff_dest = new_dest - old_dest

# Prepare DataFrame
sample = pd.DataFrame({
    "type": [trans_type],
    "amount": [amount],
    "oldbalanceOrg": [old_org],
    "newbalanceOrig": [new_org],
    "oldbalanceDest": [old_dest],
    "newbalanceDest": [new_dest],
    "balanceDiffOrig": [diff_org],
    "balanceDiffDest": [diff_dest]
})

st.subheader("Transaction Summary")
st.dataframe(sample)

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("Predict Fraud"):
    if model is None:
        st.error("❌ Model not loaded.")
    else:
        try:
            prediction = model.predict(sample)[0]
            proba = model.predict_proba(sample)[0][1] if hasattr(model, "predict_proba") else 0.0

            if prediction == 1:
                st.error(f"🚨 Fraud Detected! (Probability: {proba:.2f})")
            else:
                st.success(f"✅ Legit Transaction (Probability: {proba:.2f})")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

