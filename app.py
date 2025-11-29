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
import subprocess

st.set_page_config(page_title="Credit Card Fraud Detection Dashboard", layout="wide")
sns.set(style="whitegrid")

# --------------------------------------------------------------
# 1. DOWNLOAD DATASET FROM KAGGLE
# --------------------------------------------------------------
DATASET_NAME = "mlg-ulb/creditcardfraud"
CSV_FILE = "creditcard.csv"

st.sidebar.title("⚡ Dataset Loader")

if not os.path.exists(CSV_FILE):
    st.sidebar.warning("Downloading dataset from Kaggle… please wait.")

    os.environ['KAGGLE_USERNAME'] = st.secrets["KAGGLE_USERNAME"]
    os.environ['KAGGLE_KEY'] = st.secrets["KAGGLE_KEY"]

    # Download & unzip
    subprocess.run(
        f"kaggle datasets download -d {DATASET_NAME} --unzip",
        shell=True,
        check=True
    )
    st.sidebar.success("Dataset downloaded successfully!")

# Load dataset
df = pd.read_csv(CSV_FILE)

# --------------------------------------------------------------
# 2. LOAD MODELS
# --------------------------------------------------------------
st.sidebar.header("🔍 Choose Model for Prediction")

model_files = {
    "Logistic Regression": "logistic_regression.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl",
    "Isolation Forest": "isolation_forest.pkl",
    "Hybrid Model": "hybrid_model.pkl"
}

selected_model_name = st.sidebar.selectbox("Select Model", list(model_files.keys()))
model_path = os.path.join("models", model_files[selected_model_name])

# Load selected model
model = joblib.load(model_path)

# --------------------------------------------------------------
# 3. DASHBOARD HEADER
# --------------------------------------------------------------
st.title("💳 Credit Card Fraud Detection Dashboard")
st.markdown("Real-time analysis & fraud prediction using ML models")

# --------------------------------------------------------------
# 4. DATA OVERVIEW SECTION
# --------------------------------------------------------------
st.subheader("📊 Dataset Overview")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Transactions", len(df))
c2.metric("Fraud Cases", df["Class"].sum())
c3.metric("Fraud %", round(df["Class"].mean() * 100, 3))
c4.metric("Features", df.shape[1])

# --------------------------------------------------------------
# 5. VISUALIZATIONS
# --------------------------------------------------------------
st.subheader("📈 Visualizations")

tab1, tab2, tab3 = st.tabs(["Fraud Distribution", "Amount Distribution", "Correlation Matrix"])

# --- Fraud distribution ---
with tab1:
    fig = px.pie(
        df,
        names="Class",
        title="Fraud vs Non-Fraud",
        color="Class",
        color_discrete_map={0: "green", 1: "red"}
    )
    st.plotly_chart(fig)

# --- Amount distribution ---
with tab2:
    fig = px.histogram(df, x="Amount", nbins=100, title="Transaction Amount Distribution")
    st.plotly_chart(fig)

# --- Correlation matrix ---
with tab3:
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# --------------------------------------------------------------
# 6. MANUAL INPUT FOR PREDICTION
# --------------------------------------------------------------
st.subheader("🧮 Fraud Prediction for a Single Transaction")

col1, col2, col3 = st.columns(3)

amount = col1.number_input("Transaction Amount ($)", 0, 100000, 100)
v1 = col2.number_input("V1", value=0.0)
v2 = col3.number_input("V2", value=0.0)
v3 = col1.number_input("V3", value=0.0)
v4 = col2.number_input("V4", value=0.0)
v5 = col3.number_input("V5", value=0.0)

# Prepare input
user_data = pd.DataFrame([[v1, v2, v3, v4, v5, amount]], 
                         columns=["V1", "V2", "V3", "V4", "V5", "Amount"])

if st.button("Predict Fraud"):
    prediction = model.predict(user_data)[0]
    proba = model.predict_proba(user_data)[0][1] if hasattr(model, "predict_proba") else None

    if prediction == 1:
        st.error(f"⚠ Fraud Detected!  (Probability: {proba:.3f})")
    else:
        st.success(f"✔ Legit Transaction  (Probability: {proba:.3f})")

# --------------------------------------------------------------
# 7. SHOW RAW DATA
# --------------------------------------------------------------
st.subheader("📄 Dataset Preview")
st.dataframe(df.head(50))

st.success("App Loaded Successfully!")
