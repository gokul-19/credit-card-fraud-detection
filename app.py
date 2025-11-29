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
st.markdown("Predict fraud using uploaded CSV files or by entering single transaction details manually.")

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
# TABS FOR DASHBOARD
# ============================================================
tab1, tab2, tab3 = st.tabs(["Upload CSV", "Manual Transaction", "Visualizations"])

# ============================================================
# TAB 1: CSV Upload
# ============================================================
with tab1:
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    if uploaded_file and model:
        df = pd.read_csv(uploaded_file)
        st.subheader("Uploaded Data")
        st.dataframe(df.head())

        # Filter transactions by amount
        min_amount = float(df['Amount'].min())
        max_amount = float(df['Amount'].max())
        amount_range = st.slider("Filter by Transaction Amount", min_value=min_amount, max_value=max_amount, value=(min_amount,max_amount))
        df_filtered = df[(df['Amount'] >= amount_range[0]) & (df['Amount'] <= amount_range[1])]
        st.write(f"Filtered transactions: {df_filtered.shape[0]} rows")

        # Preprocess
        scaler = StandardScaler()
        df_filtered['Amount_scaled'] = scaler.fit_transform(df_filtered[['Amount']])
        if 'Time' in df_filtered.columns:
            df_filtered['Time_scaled'] = scaler.fit_transform(df_filtered[['Time']])

        X = df_filtered.drop(columns=['Class','Amount','Time'], errors='ignore')

        # Predict
        df_filtered['Prediction'] = model.predict(X)
        df_filtered['FraudLabel'] = df_filtered['Prediction'].map({0:'Legitimate',1:'Fraud'})
        df_filtered['FraudProbability'] = model.predict_proba(X)[:,1]*100

        # Display high-risk
        st.subheader("High-Risk Transactions (Probability > 90%)")
        high_risk = df_filtered[df_filtered['FraudProbability']>90]
        st.dataframe(high_risk[['Prediction','FraudLabel','FraudProbability']].sort_values(by='FraudProbability',ascending=False))

        # Download predictions
        csv = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions with Probabilities", csv, "fraud_predictions.csv","text/csv")

# ============================================================
# TAB 2: Manual Transaction Input
# ============================================================
with tab2:
    st.subheader("Enter Single Transaction Details")
    with st.form("transaction_form"):
        tx_type = st.selectbox("Transaction Type", ["TRANSFER","CASH_OUT","DEBIT","PAYMENT","CASH_IN"])
        oldbalance_orig = st.number_input("Sender's Old Balance ($)", min_value=0.0, value=1000.0, step=0.01)
        amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=100.0, step=0.01)
        newbalance_orig = st.number_input("Sender's New Balance ($)", min_value=0.0, value=900.0, step=0.01)
        newbalance_dest = st.number_input("Receiver's New Balance ($)", min_value=0.0, value=500.0, step=0.01)
        submitted = st.form_submit_button("Check Fraud")

    if submitted and model:
        df_input = pd.DataFrame({
            "type":[tx_type],
            "oldbalanceOrg":[oldbalance_orig],
            "newbalanceOrig":[newbalance_orig],
            "newbalanceDest":[newbalance_dest],
            "amount":[amount],
            "balanceDiffOrig":[oldbalance_orig-newbalance_orig],
            "balanceDiffDest":[newbalance_dest-0]
        })

        # Scaling
        numeric_cols = ["oldbalanceOrg","newbalanceOrig","newbalanceDest","amount","balanceDiffOrig","balanceDiffDest"]
        scaler = StandardScaler()
        df_input[numeric_cols] = scaler.fit_transform(df_input[numeric_cols])

        # One-hot encode type
        df_input = pd.get_dummies(df_input, columns=["type"], drop_first=True)

        # Add missing columns for model
        for col in model.feature_names_in_:
            if col not in df_input.columns:
                df_input[col]=0
        df_input = df_input[model.feature_names_in_]

        # Predict
        pred = model.predict(df_input)[0]
        prob = model.predict_proba(df_input)[0][1]*100

        st.subheader("Transaction Summary")
        st.write(f"Type: {tx_type}")
        st.write(f"Sender's Old Balance: ${oldbalance_orig}")
        st.write(f"Transaction Amount: ${amount}")
        st.write(f"Sender's New Balance: ${newbalance_orig}")
        st.write(f"Receiver's New Balance: ${newbalance_dest}")

        st.subheader("Fraud Prediction")
        st.write(f"Predicted Class: {'Fraud' if pred==1 else 'Legitimate'}")
        st.write(f"Fraud Probability: {prob:.2f}%")

        if prob>90:
            st.warning("⚠️ High-Risk Transaction!")
        elif pred==1:
            st.error("⚠️ Fraud Detected!")
        else:
            st.success("✅ Transaction is likely legitimate")

# ============================================================
# TAB 3: Visualizations
# ============================================================
with tab3:
    st.subheader("Visualizations")
    if uploaded_file:
        fig = px.histogram(df_filtered, x='Class', color='Class', title="Fraud vs Legit Transactions")
        st.plotly_chart(fig,use_container_width=True)

        fig2, ax2 = plt.subplots()
        sns.histplot(df_filtered['Amount'], bins=50, kde=True, ax=ax2)
        ax2.set_title("Transaction Amount Distribution")
        st.pyplot(fig2)
    else:
        st.info("Upload a CSV to see visualizations.")

