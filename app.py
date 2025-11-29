# ============================================================
# STREAMLIT FRAUD DETECTION DASHBOARD 
# ============================================================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import numpy as np

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("creditcard.csv")  # Replace with your dataset

# Feature engineering
df['balanceDiffOrig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
df['balanceDiffDest'] = df['newbalanceDest'] - df['oldbalanceDest']

# Features and target
features = ['Amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'balanceDiffOrig', 'balanceDiffDest']
X = df[features]
y = df['Class']

# -----------------------------
# Split data
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# -----------------------------
# Scale numeric features
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, "scaler.pkl")
print("Scaler saved as scaler.pkl")

# -----------------------------
# 1️⃣ Logistic Regression
# -----------------------------
logreg = LogisticRegression(class_weight='balanced', max_iter=2000, random_state=42)
logreg.fit(X_train_scaled, y_train)
joblib.dump(logreg, "logistic_regression.pkl")
print("Logistic Regression saved as logistic_regression.pkl")

# -----------------------------
# 2️⃣ Random Forest
# -----------------------------
rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)
joblib.dump(rf, "random_forest.pkl")
print("Random Forest saved as random_forest.pkl")

# -----------------------------
# 3️⃣ XGBoost
# -----------------------------
xgb = XGBClassifier(n_estimators=200, scale_pos_weight=(len(y_train)-sum(y_train))/sum(y_train), random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb.fit(X_train, y_train)
joblib.dump(xgb, "xgboost.pkl")
print("XGBoost saved as xgboost.pkl")

# -----------------------------
# 4️⃣ Isolation Forest (unsupervised)
# -----------------------------
iso = IsolationForest(n_estimators=200, contamination=0.01, random_state=42)
iso.fit(X_train)  # Use unscaled raw features
joblib.dump(iso, "isolation_forest.pkl")
print("Isolation Forest saved as isolation_forest.pkl")

# -----------------------------
# 5️⃣ Hybrid Model (RF + XGB + Isolation Forest)
# -----------------------------
# Example: simple ensemble averaging probabilities
class HybridModel:
    def __init__(self, rf_model, xgb_model, iso_model):
        self.rf = rf_model
        self.xgb = xgb_model
        self.iso = iso_model
    
    def predict(self, X):
        # RF and XGB probabilities
        prob_rf = self.rf.predict_proba(X)[:,1]
        prob_xgb = self.xgb.predict_proba(X)[:,1]
        # Isolation Forest anomaly score: -1 for outliers, 1 for inliers
        iso_scores = self.iso.predict(X)
        prob_iso = np.where(iso_scores==-1, 1, 0)  # treat anomalies as fraud
        # Average
        avg_prob = (prob_rf + prob_xgb + prob_iso) / 3
        return (avg_prob > 0.5).astype(int)
    
    def predict_proba(self, X):
        prob_rf = self.rf.predict_proba(X)[:,1]
        prob_xgb = self.xgb.predict_proba(X)[:,1]
        iso_scores = self.iso.predict(X)
        prob_iso = np.where(iso_scores==-1, 1, 0)
        avg_prob = (prob_rf + prob_xgb + prob_iso) / 3
        return np.vstack([1-avg_prob, avg_prob]).T

hybrid = HybridModel(rf, xgb, iso)
joblib.dump(hybrid, "hybrid_model.pkl")
print("Hybrid Model saved as hybrid_model.pkl")
