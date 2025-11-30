import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from io import BytesIO

# Streamlit page config
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection Dashboard")

# Google Drive model link
FILE_ID = "1Hjxc5wS13dMRWJkNUhRLEXRPBo5tga0Z"
MODEL_URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

@st.cache_data
def load_model_from_drive():
    try:
        response = requests.get(MODEL_URL)
        response.raise_for_status()
        model = joblib.load(BytesIO(response.content))
        return model
    except Exception as e:
        st.error(f"Failed to download/load model: {e}")
        return None

model = load_model_from_drive()
if model:
    st.success("✅ Model loaded successfully from Drive!")
else:
    st.warning("⚠️ Model Not Loaded. Please check link or internet connection.")

# Input form
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

diff_org = old_org - new_org
diff_dest = new_dest - old_dest

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

# Prediction
if st.button("Predict Fraud"):
    if model is None:
        st.error("❌ Model not loaded.")
    else:
        try:
            pred = model.predict(sample)[0]
            proba = model.predict_proba(sample)[0][1] if hasattr(model, "predict_proba") else 0.0

            if pred == 1:
                st.error(f"🚨 Fraud Detected! (Probability: {proba:.2f})")
            else:
                st.success(f"✅ Legit Transaction (Probability: {proba:.2f})")
        except Exception as e:
            st.error(f"Prediction Error: {e}")


