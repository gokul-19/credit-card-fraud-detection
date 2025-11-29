# ============================================================
# STREAMLIT FRAUD DETECTION DASHBOARD
# ============================================================
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler

sns.set(style="whitegrid")

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection Dashboard")
st.markdown("Upload a CSV file to detect fraudulent transactions using a trained Random Forest model.")

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model():
    try:
        model = joblib.load("random_forest.pkl")
        return model
    except:
        st.error("Random Forest model not found. Make sure random_forest.pkl is in the folder!")
        return None

model = load_model()

# ============================================================
# FILE UPLOAD
# ============================================================
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file and model:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data")
    st.dataframe(df.head())

    # ============================================================
    # SHOW DATA INFO
    # ============================================================
    st.subheader("Dataset Summary")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])
    with col3:
        if 'Class' in df.columns:
            st.metric("Fraud %", f"{round(df['Class'].mean()*100,2)}%")

    # ============================================================
    # VISUALIZATIONS
    # ============================================================
    st.subheader("Visualizations")
    if 'Class' in df.columns:
        fig, ax = plt.subplots()
        sns.countplot(data=df, x='Class', palette='coolwarm', ax=ax)
        ax.set_title("Fraud vs Non-Fraud")
        st.pyplot(fig)

    if 'Amount' in df.columns:
        fig2, ax2 = plt.subplots()
        sns.histplot(df['Amount'], bins=50, kde=True, ax=ax2)
        ax2.set_title("Transaction Amount Distribution")
        st.pyplot(fig2)

    # ============================================================
    # PREPROCESS DATA FOR PREDICTION
    # ============================================================
    st.subheader("Fraud Predictions")
    try:
        # Scale Amount and Time
        if 'Amount_scaled' not in df.columns:
            scaler = StandardScaler()
            if 'Amount' in df.columns:
                df['Amount_scaled'] = scaler.fit_transform(df[['Amount']])
            if 'Time' in df.columns:
                df['Time_scaled'] = scaler.fit_transform(df[['Time']])

        # Prepare features (drop columns not used)
        X = df.drop(columns=['Class', 'Amount', 'Time'], errors='ignore')

        # Make predictions
        df['Prediction'] = model.predict(X)
        df['FraudLabel'] = df['Prediction'].map({0: 'Legitimate', 1: 'Fraud'})

        st.success("Predictions completed!")
        st.dataframe(df[['Prediction', 'FraudLabel']].head(20))

        # Download predictions
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Predictions",
            csv,
            "fraud_predictions.csv",
            "text/csv"
        )

    except Exception as e:
        st.error(f"Error during prediction: {e}")

else:
    st.info("Please upload a CSV file to start predictions.")
