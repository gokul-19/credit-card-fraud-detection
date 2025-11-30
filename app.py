import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import gdown
import joblib
import os

# =========================================================
# STREAMLIT CONFIG
# =========================================================
st.set_page_config(
    page_title="Credit Card Fraud Detection Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# GOOGLE DRIVE DOWNLOAD
# =========================================================
# Your shared link
FILE_ID = "1eRNEgQKTAOC51zPdhXQcgzryXLk7QbVA"
DATA_URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

def load_dataset():
    """Downloads dataset if missing"""
    if not os.path.exists("dataset.csv"):
        try:
            st.info("📥 Downloading dataset from Google Drive ...")
            gdown.download(DATA_URL, "dataset.csv", quiet=False)
        except Exception as e:
            st.error(f"Download Error: {e}")
            return None
    
    try:
        df = pd.read_csv("dataset.csv")
        return df
    except Exception as e:
        st.error(f"CSV Read Error: {e}")
        return None

@st.cache_data
def get_data():
    return load_dataset()

df = get_data()

if df is None:
    st.stop()

# =========================================================
# LOAD MODELS
# =========================================================
def load_model(path):
    try:
        return joblib.load(path)
    except:
        return None

models = {
    "Logistic Regression": load_model("models/logistic_regression.pkl"),
    "Random Forest": load_model("models/random_forest.pkl"),
    "XGBoost": load_model("models/xgboost_model.pkl"),
    "Isolation Forest": load_model("models/isolation_forest.pkl")
}

# =========================================================
# SIDEBAR NAVIGATION
# =========================================================
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "📊 Dataset Overview",
        "📈 Data Visualizations",
        "🤖 Model Comparison",
        "🔍 Predict Fraud"
    ]
)

# =========================================================
# PAGE: HOME
# =========================================================
if page == "🏠 Home":
    st.title("💳 Credit Card Fraud Detection Dashboard")
    st.markdown("""
    ### 🔍 Overview  
    This dashboard allows:
    - Dataset Exploration  
    - Visual Fraud Analysis  
    - Model Performance Comparison  
    - Single Transaction Fraud Prediction  

    **Models included:**
    - Logistic Regression  
    - Random Forest  
    - XGBoost  
    - Isolation Forest  
    """)

# =========================================================
# PAGE: DATASET OVERVIEW
# =========================================================
elif page == "📊 Dataset Overview":
    st.title("📊 Dataset Overview")

    st.subheader("First 5 Rows")
    st.dataframe(df.head())

    st.subheader("Dataset Shape")
    st.write(df.shape)

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    st.subheader("Fraud Distribution")
    fig = px.pie(df, names="isFraud", title="Fraud vs Non-Fraud")
    st.plotly_chart(fig)

# =========================================================
# PAGE: VISUALIZATIONS
# =========================================================
elif page == "📈 Data Visualizations":
    st.title("📈 Exploratory Visualizations")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Transaction Types")
        fig1 = px.bar(df["type"].value_counts(), title="Transaction Types")
        st.plotly_chart(fig1)

    with col2:
        st.subheader("Fraud Rate by Type")
        fraud_rate = df.groupby("type")["isFraud"].mean()
        fig2 = px.bar(fraud_rate, title="Fraud Rate by Type")
        st.plotly_chart(fig2)

    st.subheader("Amount Distribution (Log)")
    df["log_amount"] = np.log1p(df["amount"])
    fig3 = px.histogram(df, x="log_amount", nbins=100)
    st.plotly_chart(fig3)

    st.subheader("Correlation Heatmap")
    corr_cols = ["amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest","isFraud"]
    fig4 = px.imshow(df[corr_cols].corr(), text_auto=True)
    st.plotly_chart(fig4)

# =========================================================
# PAGE: MODEL COMPARISON
# =========================================================
elif page == "🤖 Model Comparison":
    st.title("🤖 Model Performance Comparison")

    for name, model in models.items():
        if model is None:
            st.error(f"{name}: ❌ Not Found")
        else:
            st.success(f"{name}: ✔ Loaded Successfully")

    st.info("Upload metrics screenshot or text if you want to display full comparison.")

# =========================================================
# PAGE: SINGLE PREDICTION
# =========================================================
elif page == "🔍 Predict Fraud":
    st.title("🔍 Predict Single Transaction Fraud")

    st.write("Enter transaction details below:")

    col1, col2, col3 = st.columns(3)

    with col1:
        t_type = st.selectbox("Transaction Type", df["type"].unique())
        amount = st.number_input("Amount", min_value=0.0)

    with col2:
        old_org = st.number_input("Old Balance (Sender)", min_value=0.0)
        new_org = st.number_input("New Balance (Sender)", min_value=0.0)

    with col3:
        old_dest = st.number_input("Old Balance (Receiver)", min_value=0.0)
        new_dest = st.number_input("New Balance (Receiver)", min_value=0.0)

    # Derived features
    diff_org = old_org - new_org
    diff_dest = new_dest - old_dest

    sample = pd.DataFrame({
        "type": [t_type],
        "amount": [amount],
        "oldbalanceOrg": [old_org],
        "newbalanceOrig": [new_org],
        "oldbalanceDest": [old_dest],
        "newbalanceDest": [new_dest],
        "balanceDiffOrig": [diff_org],
        "balanceDiffDest": [diff_dest]
    })

    st.subheader("Input Summary")
    st.dataframe(sample)

    model_choice = st.selectbox("Choose Model", list(models.keys()))

    if st.button("Predict"):
        model = models[model_choice]

        if model is None:
            st.error("Model not found!")
        else:
            pred = model.predict(sample)[0]

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(sample)[0][1]
            else:
                proba = 0.00

            if pred == 1:
                st.error(f"🚨 Fraud Detected! (Prob: {proba:.2f})")
            else:
                st.success(f"✅ Legitimate Transaction (Prob: {proba:.2f})")


