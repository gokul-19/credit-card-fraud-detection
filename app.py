# ============================================================
# STREAMLIT FRAUD DETECTION DASHBOARD 
# ============================================================
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
CSV_URL = "https://drive.google.com/uc?id=1Kwm31irBeeMqRmyp6fumkGSxlnGSmrzY"

@st.cache_data
def load_data():
    return pd.read_csv(CSV_URL)

df = load_data()

# Rename "Class" to "isFraud" (Kaggle naming)
if "Class" in df.columns:
    df.rename(columns={"Class": "isFraud"}, inplace=True)

# ------------------------------------------------------------------
# 2. LOAD ALL ML MODELS
# ------------------------------------------------------------------
model_files = {
    "Logistic Regression": "models/logistic_regression.pkl",
    "Random Forest": "models/random_forest.pkl",
    "XGBoost": "models/xgboost.pkl",
    "Isolation Forest": "models/isolation_forest.pkl",
    "Hybrid Model": "models/hybrid_model.pkl"
}

st.sidebar.header("🔍 Select Fraud Detection Model")
selected_model_name = st.sidebar.selectbox("Choose Model", list(model_files.keys()))
model = joblib.load(model_files[selected_model_name])

# ------------------------------------------------------------------
# 3. HEADER
# ------------------------------------------------------------------
st.title("💳 Credit Card Fraud Detection Dashboard")
st.markdown("Real-time fraud detection using machine learning models. Powered by Streamlit.")

# ------------------------------------------------------------------
# 4. SUMMARY CARDS
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

# Fraud distribution
with tab1:
    fig = px.pie(df, names="isFraud", title="Fraud vs Non-Fraud", color="isFraud")
    st.plotly_chart(fig)

# Amount Distribution
with tab2:
    fig = px.histogram(df, x="Amount", nbins=50, title="Transaction Amount Distribution")
    st.plotly_chart(fig)

# Correlation Heatmap
with tab3:
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ------------------------------------------------------------------
# 6. FRAUD PREDICTION (Single Transaction)
# ------------------------------------------------------------------
st.subheader("🧮 Predict Fraud for a Single Transaction")

col1, col2, col3 = st.columns(3)

amount = col1.number_input("Transaction Amount ($)", 0, 50000, 100)
v1 = col2.number_input("V1", value=0.0)
v2 = col3.number_input("V2", value=0.0)
v3 = col1.number_input("V3", value=0.0)
v4 = col2.number_input("V4", value=0.0)
v5 = col3.number_input("V5", value=0.0)

# Prepare data
input_data = pd.DataFrame([[v1, v2, v3, v4, v5, amount]],
                          columns=["V1", "V2", "V3", "V4", "V5", "Amount"])

if st.button("Predict Fraud"):
    prediction = model.predict(input_data)[0]

    # probability if available
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_data)[0][1]
    else:
        probability = 0.0

    if prediction == 1:
        st.error(f"⚠ Fraud Detected! (Probability: {probability:.4f})")
    else:
        st.success(f"✔ Legit Transaction (Probability: {probability:.4f})")

# ------------------------------------------------------------------
# 7. SHOW RAW DATA
# ------------------------------------------------------------------
st.subheader("📄 Dataset Preview")
st.dataframe(df.head(50))

st.success("Dashboard loaded successfully!")

