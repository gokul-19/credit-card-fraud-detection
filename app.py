import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import gdown
import joblib
import os

# -----------------------------------------------------------
# STREAMLIT CONFIG
# -----------------------------------------------------------
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("💳 Credit Card Fraud Detection Dashboard (FAST MODE)")

# -----------------------------------------------------------
# GOOGLE DRIVE DATASET LINK
# -----------------------------------------------------------
FILE_ID = "1eRNEgQKTAOC51zPdhXQcgzryXLk7QbVA"
DATA_URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

DATA_DIR = "data"
DATA_PATH = "data/dataset.csv"

# -----------------------------------------------------------
# DOWNLOAD DATA ONLY ONCE
# -----------------------------------------------------------
def download_once():
    """Download dataset only if missing (500 MB)."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    if not os.path.exists(DATA_PATH):
        st.info("📥 Downloading dataset (first time only)… Please wait 10–20 seconds.")
        gdown.download(DATA_URL, DATA_PATH, quiet=False)
        st.success("Dataset downloaded!")

@st.cache_data(show_spinner=True)
def load_sample():
    """Load a smaller sample for fast UI (10%)."""
    df = pd.read_csv(DATA_PATH)
    sample = df.sample(frac=0.10, random_state=42)
    return df, sample

# -----------------------------------------------------------
# INITIALIZE
# -----------------------------------------------------------
download_once()

st.success("Dataset found ✔ Loading preview…")

df, sample_df = load_sample()

# -----------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------
st.sidebar.header("📌 Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["🏠 Home", "📊 Data Overview", "📈 Visualizations", "🤖 Models", "🔍 Predict Fraud"]
)

# -----------------------------------------------------------
# LOAD MODELS
# -----------------------------------------------------------
def load_model(file):
    try:
        return joblib.load(f"models/{file}")
    except:
        return None

models = {
    "Logistic Regression": load_model("logistic_regression.pkl"),
    "Random Forest": load_model("random_forest.pkl"),
    "XGBoost": load_model("xgboost_model.pkl"),
    "Isolation Forest": load_model("isolation_forest.pkl")
}

# ===========================================================
# PAGE: HOME
# ===========================================================
if page == "🏠 Home":
    st.subheader("Welcome!")

    st.markdown("""
    ### 🔍 What this dashboard offers:
    - Fast dataset preview (10% sample)
    - Full dataset used for ML predictions
    - Multiple ML models supported
    - Smooth and fast performance on Streamlit Cloud  
    """)

# ===========================================================
# PAGE: DATA OVERVIEW
# ===========================================================
elif page == "📊 Data Overview":
    st.header("📊 Dataset Overview")
    st.write(sample_df.head())
    st.write("Shape:", sample_df.shape)

    st.subheader("Fraud distribution")
    fig = px.pie(sample_df, names="isFraud", title="Fraud vs Non-Fraud")
    st.plotly_chart(fig)

# ===========================================================
# PAGE: VISUALIZATIONS
# ===========================================================
elif page == "📈 Visualizations":
    st.header("📈 Visualizations (Sampled Data for Speed)")

    st.subheader("Transaction Types")
    fig1 = px.bar(sample_df['type'].value_counts(), title="Transaction Types")
    st.plotly_chart(fig1)

    st.subheader("Fraud Rate by Type")
    fraud_rate = sample_df.groupby("type")["isFraud"].mean()
    fig2 = px.bar(fraud_rate, title="Fraud Rate by Type")
    st.plotly_chart(fig2)

    st.subheader("Amount Distribution (Log Scale)")
    sample_df["log_amount"] = np.log1p(sample_df["amount"])
    fig3 = px.histogram(sample_df, x="log_amount", nbins=80, title="Log Amount Distribution")
    st.plotly_chart(fig3)

# ===========================================================
# PAGE: MODEL CHECK
# ===========================================================
elif page == "🤖 Models":
    st.header("🤖 Model Status")

    for name, model in models.items():
        if model:
            st.success(f"{name}: Loaded ✔")
        else:
            st.error(f"{name}: Not Found ❌")

# ===========================================================
# PAGE: PREDICTION
# ===========================================================
elif page == "🔍 Predict Fraud":
    st.header("🔍 Predict Fraud for a Transaction")

    col1, col2, col3 = st.columns(3)

    with col1:
        trans_type = st.selectbox("Transaction Type", df["type"].unique())
        amount = st.number_input("Amount", 0.0)

    with col2:
        old_org = st.number_input("Old Sender Balance", 0.0)
        new_org = st.number_input("New Sender Balance", 0.0)

    with col3:
        old_dest = st.number_input("Old Receiver Balance", 0.0)
        new_dest = st.number_input("New Receiver Balance", 0.0)

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

    st.subheader("Input Summary")
    st.dataframe(sample)

    model_choice = st.selectbox("Choose Model", list(models.keys()))

    if st.button("Predict"):
        model = models[model_choice]

        if model is None:
            st.error("Model missing!")
        else:
            pred = model.predict(sample)[0]
            proba = model.predict_proba(sample)[0][1] if hasattr(model, "predict_proba") else 0.0

            if pred == 1:
                st.error(f"🚨 Fraud Detected! (Prob: {proba:.2f})")
            else:
                st.success(f"✅ Legit Transaction (Prob: {proba:.2f})")


