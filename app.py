import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# ------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection Dashboard",
    layout="wide"
)
sns.set(style="whitegrid")

# ------------------------------------------------------------------
# 1. LOAD DATA FROM GOOGLE DRIVE
# ------------------------------------------------------------------
CSV_URL = "https://drive.google.com/uc?id=1eRNEgQKTAOC51zPdhXQcgzryXLk7QbVA"

@st.cache_data
def load_data():
    return pd.read_csv(CSV_URL)

df = load_data()

# Ensure 'isFraud' exists
if "isFraud" not in df.columns:
    st.error("Dataset does not have 'isFraud' column.")
    st.stop()

# ------------------------------------------------------------------
# 2. DYNAMIC MODEL LOADER
# ------------------------------------------------------------------
MODELS_DIR = "models"

available_models = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pkl")]

if not available_models:
    st.error("No model files found in models/ folder! Please upload .pkl files.")
    st.stop()

selected_model_file = st.sidebar.selectbox("🔍 Select Model", available_models)
model_path = os.path.join(MODELS_DIR, selected_model_file)
model = joblib.load(model_path)
st.sidebar.success(f"Loaded model: {selected_model_file}")

# ------------------------------------------------------------------
# 3. DASHBOARD HEADER
# ------------------------------------------------------------------
st.title("💳 Credit Card Fraud Detection Dashboard")
st.markdown("Real-time analysis & fraud prediction using available ML models.")

# ------------------------------------------------------------------
# 4. DATA SUMMARY CARDS
# ------------------------------------------------------------------
st.subheader("📊 Dataset Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Transactions", len(df))
c2.metric("Fraud Cases", int(df["isFraud"].sum()))
c3.metric("Fraud %", round(df["isFraud"].mean() * 100, 3))
c4.metric("Features", df.shape[1])

# ------------------------------------------------------------------
# 5. VISUALIZATIONS
# ------------------------------------------------------------------
st.subheader("📈 Visualizations")
tab1, tab2, tab3 = st.tabs(["Fraud Distribution", "Amount Histogram", "Correlation Matrix"])

with tab1:
    fig = px.pie(df, names="isFraud", title="Fraud vs Non-Fraud", color="isFraud")
    st.plotly_chart(fig)

with tab2:
    fig = px.histogram(df, x="amount", nbins=50, title="Transaction Amount Distribution")
    st.plotly_chart(fig)

with tab3:
    corr = df[["amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest","isFraud"]].corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ------------------------------------------------------------------
# 6. SINGLE TRANSACTION PREDICTION
# ------------------------------------------------------------------
st.subheader("🧮 Predict Fraud for a Single Transaction")

col1, col2, col3 = st.columns(3)
amount = col1.number_input("Transaction Amount ($)", 0, 50000, 100)
oldbalanceOrg = col2.number_input("Sender's Old Balance ($)", 0, 1000000, 0)
newbalanceOrig = col3.number_input("Sender's New Balance ($)", 0, 1000000, 0)
oldbalanceDest = col1.number_input("Receiver's Old Balance ($)", 0, 1000000, 0)
newbalanceDest = col2.number_input("Receiver's New Balance ($)", 0, 1000000, 0)

# For simplicity, we use numeric columns for prediction
input_data = pd.DataFrame([[amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest]],
                          columns=["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"])

if st.button("Predict Fraud"):
    try:
        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1] if hasattr(model, "predict_proba") else 0.0
        if pred == 1:
            st.error(f"⚠ Fraud Detected! (Probability: {prob:.4f})")
        else:
            st.success(f"✔ Legit Transaction (Probability: {prob:.4f})")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ------------------------------------------------------------------
# 7. SHOW RAW DATA
# ------------------------------------------------------------------
st.subheader("📄 Dataset Preview")
st.dataframe(df.head(50))

st.success("Dashboard loaded successfully!")

