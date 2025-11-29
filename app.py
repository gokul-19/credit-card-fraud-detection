# ============================================================
# STREAMLIT FRAUD DETECTION DASHBOARD (SHAP REMOVED)
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler

sns.set(style="whitegrid")

st.set_page_config(
    page_title="Credit Card Fraud Detection Dashboard",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Card Fraud Detection Dashboard")
st.markdown("Upload a CSV file to detect fraudulent transactions using a trained Random Forest model.")

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_model(path="random_forest.pkl"):
    try:
        model = joblib.load(path)
        return model
    except:
        st.error(f"Model file {path} not found!")
        return None

model = load_model()

# ============================================================
# FILE UPLOAD
# ============================================================
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file and model:
    df = pd.read_csv(uploaded_file)

    # ============================================================
    # TABS
    # ============================================================
    tab1, tab2, tab3 = st.tabs(["Dataset Overview", "Visualizations", "Predictions"])

    # ============================================================
    # TAB 1: Dataset Overview
    # ============================================================
    with tab1:
        st.subheader("📌 Uploaded Data")
        st.dataframe(df.head())

        st.markdown("### Summary Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Transactions", df.shape[0])
        col2.metric("Total Fraud", int(df.get('Class', pd.Series([0]*len(df))).sum()))
        col3.metric("Average Amount", round(df['Amount'].mean(), 2))
        col4.metric("Max Amount", round(df['Amount'].max(), 2))

        st.markdown("### Filters")
        min_amount = float(df['Amount'].min())
        max_amount = float(df['Amount'].max())
        amount_range = st.slider("Transaction Amount Range", min_value=min_amount, max_value=max_amount, value=(min_amount, max_amount))
        df_filtered = df[(df['Amount'] >= amount_range[0]) & (df['Amount'] <= amount_range[1])]
        st.write(f"Filtered transactions: {df_filtered.shape[0]} rows")

    # ============================================================
    # TAB 2: Visualizations
    # ============================================================
    with tab2:
        st.subheader("📊 Visualizations")
        if 'Class' in df_filtered.columns:
            fig = px.histogram(df_filtered, x='Class', color='Class', title="Fraud vs Legit Transactions")
            st.plotly_chart(fig, use_container_width=True)

        fig2, ax2 = plt.subplots()
        sns.histplot(df_filtered['Amount'], bins=50, kde=True, ax=ax2)
        ax2.set_title("Transaction Amount Distribution")
        st.pyplot(fig2)

    # ============================================================
    # TAB 3: Predictions
    # ============================================================
    with tab3:
        st.subheader("🔍 Fraud Predictions")
        try:
            # Scale Amount & Time
            scaler = StandardScaler()
            if 'Amount_scaled' not in df_filtered.columns:
                df_filtered['Amount_scaled'] = scaler.fit_transform(df_filtered[['Amount']])
            if 'Time_scaled' not in df_filtered.columns and 'Time' in df_filtered.columns:
                df_filtered['Time_scaled'] = scaler.fit_transform(df_filtered[['Time']])

            X = df_filtered.drop(columns=['Class', 'Amount', 'Time'], errors='ignore')

            # Predictions
            df_filtered['Prediction'] = model.predict(X)
            df_filtered['FraudLabel'] = df_filtered['Prediction'].map({0: 'Legitimate', 1: 'Fraud'})
            df_filtered['FraudProbability'] = model.predict_proba(X)[:,1]*100

            # Highlight high-risk transactions (>90%)
            st.write("### High-Risk Transactions (Fraud Probability > 90%)")
            high_risk = df_filtered[df_filtered['FraudProbability'] > 90]
            st.dataframe(high_risk[['Prediction','FraudLabel','FraudProbability']].sort_values(by='FraudProbability', ascending=False))

            # ============================================================
            # Download predictions
            # ============================================================
            csv = df_filtered.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Predictions with Probabilities", csv, "fraud_predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"Error during prediction: {e}")

else:
    st.info("Please upload a CSV file to start predictions.")

