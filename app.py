# ============================================================
# STREAMLIT FRAUD DETECTION DASHBOARD (SHAP REMOVED)
# ============================================================
import streamlit as st
import pandas as pd
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
st.markdown("Upload a CSV file to detect fraudulent transactions using a trained Random Forest model with enhanced visualizations.")

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
# TABS
# ============================================================
tab1, tab2 = st.tabs(["Upload CSV & Predictions", "Visualizations"])

# ============================================================
# TAB 1: CSV UPLOAD AND PREDICTIONS
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

        # Preprocess numeric features
        scaler = StandardScaler()
        df_filtered['Amount_scaled'] = scaler.fit_transform(df_filtered[['Amount']])
        if 'Time' in df_filtered.columns:
            df_filtered['Time_scaled'] = scaler.fit_transform(df_filtered[['Time']])

        # Prepare input for model
        X = df_filtered.drop(columns=['Class','Amount','Time'], errors='ignore')

        # Predict fraud
        df_filtered['Prediction'] = model.predict(X)
        df_filtered['FraudLabel'] = df_filtered['Prediction'].map({0:'Legitimate',1:'Fraud'})
        df_filtered['FraudProbability'] = model.predict_proba(X)[:,1]*100

        # Highlight high-risk transactions (>90%)
        st.subheader("High-Risk Transactions (Fraud Probability > 90%)")
        high_risk = df_filtered[df_filtered['FraudProbability']>90]
        st.dataframe(high_risk[['Prediction','FraudLabel','FraudProbability']].sort_values(by='FraudProbability',ascending=False))

        # Download predictions
        csv = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Predictions with Probabilities", csv, "fraud_predictions.csv","text/csv")

# ============================================================
# TAB 2: VISUALIZATIONS
# ============================================================
with tab2:
    st.subheader("Enhanced Visualizations")

    if uploaded_file:
        # Fraud distribution bar chart
        fig_bar = px.histogram(df_filtered, x='Class', color='Class', title="Fraud vs Legit Transactions")
        st.plotly_chart(fig_bar, use_container_width=True)

        # Fraud proportion pie chart
        fig_pie = px.pie(df_filtered, names='Class', title="Fraud vs Legit Transactions (%)", color='Class')
        st.plotly_chart(fig_pie, use_container_width=True)

        # Transaction Amount Distribution Histogram
        fig_hist, ax_hist = plt.subplots()
        sns.histplot(df_filtered['Amount'], bins=50, kde=True, ax=ax_hist)
        ax_hist.set_title("Transaction Amount Distribution")
        st.pyplot(fig_hist)

        # Box plot of Amount by Class
        fig_box, ax_box = plt.subplots()
        sns.boxplot(x='Class', y='Amount', data=df_filtered, ax=ax_box)
        ax_box.set_title('Transaction Amount by Class')
        st.pyplot(fig_box)

        # Fraud probability distribution
        fig_prob, ax_prob = plt.subplots()
        sns.histplot(df_filtered['FraudProbability'], bins=50, kde=True, color='red', ax=ax_prob)
        ax_prob.set_title('Fraud Probability Distribution')
        st.pyplot(fig_prob)

        # Correlation heatmap
        corr_cols = ['Amount_scaled']
        if 'Time_scaled' in df_filtered.columns:
            corr_cols.append('Time_scaled')
        corr_cols.append('FraudProbability')
        fig_corr, ax_corr = plt.subplots(figsize=(8,6))
        sns.heatmap(df_filtered[corr_cols].corr(), annot=True, cmap='coolwarm', ax=ax_corr)
        ax_corr.set_title('Feature Correlation Heatmap')
        st.pyplot(fig_corr)

        # Top 10 fraud transactions by Amount
        if 'TransactionID' in df_filtered.columns:
            top_fraud = df_filtered[df_filtered['Class']==1].nlargest(10,'Amount')
            fig_top, ax_top = plt.subplots()
            sns.barplot(x='Amount', y='TransactionID', data=top_fraud, ax=ax_top, palette='Reds_r')
            ax_top.set_title('Top 10 Fraud Transactions by Amount')
            st.pyplot(fig_top)
    else:
        st.info("Upload a CSV to see visualizations.")

