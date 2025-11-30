import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# -------------------------------
# STREAMLIT CONFIG
# -------------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("💳 Credit Card Fraud Detection Dashboard")

# -------------------------------
# MODEL DOWNLOAD / LOAD
# -------------------------------
# Use your own pre-trained Random Forest model
MODEL_FILE = "random_forest.pkl"

def load_model():
    if not os.path.exists("models"):
        os.makedirs("models")
    if not os.path.exists(f"models/{MODEL_FILE}"):
        st.warning(f"⚠️ {MODEL_FILE} not found. Please upload it in models folder.")
        return None
    return joblib.load(f"models/{MODEL_FILE}")

model = load_model()

# -------------------------------
# INPUT FORM
# -------------------------------
st.header("🔍 Predict Fraud for a Transaction")

col1, col2, col3 = st.columns(3)

with col1:
    trans_type = st.selectbox("Transaction Type", ["PAYMENT","TRANSFER","CASH_OUT","CASH_IN","DEBIT","OTHER"])
    amount = st.number_input("Transaction Amount ($)", 0.0)

with col2:
    old_org = st.number_input("Sender Old Balance ($)", 0.0)
    new_org = st.number_input("Sender New Balance ($)", 0.0)

with col3:
    old_dest = st.number_input("Receiver Old Balance ($)", 0.0)
    new_dest = st.number_input("Receiver New Balance ($)", 0.0)

# Compute balance differences
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

# -------------------------------
# PREDICTION
# -------------------------------
if st.button("Predict"):
    if model is None:
        st.error("Model is missing! Upload random_forest.pkl in /models folder")
    else:
        # One-hot encoding for type
        sample_encoded = pd.get_dummies(sample, columns=["type"])
        # Align columns with training data
        model_columns = model.feature_names_in_
        for col in model_columns:
            if col not in sample_encoded.columns:
                sample_encoded[col] = 0
        sample_encoded = sample_encoded[model_columns]

        pred = model.predict(sample_encoded)[0]
        proba = model.predict_proba(sample_encoded)[0][1] if hasattr(model, "predict_proba") else 0.0

        if pred == 1:
            st.error(f"🚨 Fraud Detected! (Probability: {proba:.2f})")
        else:
            st.success(f"✅ Legit Transaction (Probability: {proba:.2f})")


