import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import gdown
import joblib
import os
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Credit Card Fraud Detection Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# GOOGLE DRIVE DATA LOADING
# ---------------------------------------------------------
FILE_ID = "1eRNEgQKTAOC51zPdhXQcgzryXLk7QbVA"
DATA_URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

def load_data():
    if not os.path.exists("dataset.csv"):
        gdown.download(DATA_URL, "dataset.csv", quiet=False)
    df = pd.read_csv("dataset.csv")
    return df

@st.cache_data
def cached_data():
    return load_data()

df = cached_data()

# ---------------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------------
def load_model(filename):
    try:
        return joblib.load(filename)
    except:
        return None

models = {
    "Logistic Regression": load_model("logistic_regression.pkl"),
    "Random Forest": load_model("random_forest.pkl"),
    "XGBoost": load_model("xgboost_model.pkl"),
    "Isolation Forest": load_model("isolation_forest.pkl")
}

# ---------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "📊 Dataset Overview",
        "📈 Data Visualizations",
        "🤖 Model Performance Comparison",
        "🔍 Predict Fraud (Single Transaction)"
    ]
)

# ---------------------------------------------------------
# PAGE 1 — HOME
# ---------------------------------------------------------
if page == "🏠 Home":
    st.title("💳 Credit Card Fraud Detection Dashboard")
    st.markdown("""
    ### 🔍 Overview  
    This professional dashboard allows you to:
    - Explore the dataset  
    - Visualize fraud patterns  
    - Compare multiple ML models  
    - Predict fraud for single transactions  

    **Models included**
    - Logistic Regression  
    - Random Forest  
    - XGBoost  
    - Isolation Forest (unsupervised)  
    """)

# ---------------------------------------------------------
# PAGE 2 — DATASET OVERVIEW
# ---------------------------------------------------------
elif page == "📊 Dataset Overview":
    st.title("📊 Dataset Overview")
    st.write(df.head())
    st.write("Shape:", df.shape)

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    st.subheader("Class Distribution")
    st.write(df["isFraud"].value_counts())

    fig = px.pie(
        df,
        names="isFraud",
        title="Fraud vs Non-Fraud Distribution",
        hole=0.4
    )
    st.plotly_chart(fig)

# ---------------------------------------------------------
# PAGE 3 — VISUALIZATIONS
# ---------------------------------------------------------
elif page == "📈 Data Visualizations":
    st.title("📈 Visualizations")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Transaction Types Count")
        fig1 = px.bar(df["type"].value_counts(), title="Transaction Type Distribution")
        st.plotly_chart(fig1)

    with col2:
        st.subheader("Fraud Rate by Type")
        fraud_rate = df.groupby("type")["isFraud"].mean()
        fig2 = px.bar(fraud_rate, title="Fraud Rate by Transaction Type")
        st.plotly_chart(fig2)

    st.subheader("Amount Distribution (Log Scale)")
    df["log_amount"] = np.log1p(df["amount"])
    fig3 = px.histogram(df, x="log_amount", nbins=100, title="Log Amount Distribution")
    st.plotly_chart(fig3)

    st.subheader("Correlation Heatmap")
    corr_cols = ["amount","oldbalanceOrg","newbalanceOrig",
                 "oldbalanceDest","newbalanceDest","isFraud"]
    fig4 = px.imshow(df[corr_cols].corr(), text_auto=True, title="Correlation Matrix")
    st.plotly_chart(fig4)

# ---------------------------------------------------------
# PAGE 4 — MODEL PERFORMANCE COMPARISON
# ---------------------------------------------------------
elif page == "🤖 Model Performance Comparison":
    st.title("🤖 Model Performance Comparison")

    for model_name, model in models.items():
        if model is not None:
            st.success(f"{model_name} ✔️ Loaded")
        else:
            st.error(f"{model_name} ❌ NOT found")

    st.info("Upload your evaluation metrics section if you want full comparison.")

# ---------------------------------------------------------
# PAGE 5 — PREDICT FRAUD (SINGLE INPUT)
# ---------------------------------------------------------
elif page == "🔍 Predict Fraud (Single Transaction)":
    st.title("🔍 Single Transaction Fraud Prediction")

    st.write("Enter transaction details below:")

    col1, col2, col3 = st.columns(3)

    with col1:
        trans_type = st.selectbox("Transaction Type", df["type"].unique())
        amount = st.number_input("Amount", min_value=0.0)

    with col2:
        old_org = st.number_input("Old Balance (Sender)", min_value=0.0)
        new_org = st.number_input("New Balance (Sender)", min_value=0.0)

    with col3:
        old_dest = st.number_input("Old Balance (Receiver)", min_value=0.0)
        new_dest = st.number_input("New Balance (Receiver)", min_value=0.0)

    balanceDiffOrig = old_org - new_org
    balanceDiffDest = new_dest - old_dest

    sample = pd.DataFrame({
        "type": [trans_type],
        "amount": [amount],
        "oldbalanceOrg": [old_org],
        "newbalanceOrig": [new_org],
        "oldbalanceDest": [old_dest],
        "newbalanceDest": [new_dest],
        "balanceDiffOrig": [balanceDiffOrig],
        "balanceDiffDest": [balanceDiffDest]
    })

    st.write("### Input Summary")
    st.dataframe(sample)

    model_choice = st.selectbox(
        "Choose Model for Prediction",
        list(models.keys())
    )

    if st.button("Predict Fraud"):
        model = models[model_choice]

        if model is None:
            st.error("Model file not found!")
        else:
            prediction = model.predict(sample)[0]
            probability = model.predict_proba(sample)[0][1]

            if prediction == 1:
                st.error(f"🚨 Fraud Detected! (Probability: {probability:.2f})")
            else:
                st.success(f"✅ Legitimate Transaction (Probability: {probability:.2f})")

