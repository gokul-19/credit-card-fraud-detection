# --------------------------------------------
# 1. IMPORT LIBRARIES
# --------------------------------------------
import pandas as pd
import numpy as np
import gdown
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

# ---------------------------------------------------------
# 1. DOWNLOAD DATASET FROM GOOGLE DRIVE
# ---------------------------------------------------------
file_id = "1eRNEgQKTAOC51zPdhXQcgzryXLk7QbVA"
url = f"https://drive.google.com/uc?export=download&id={file_id}"
output = "credit_fraud_dataset.csv"

print("Downloading dataset from Google Drive...")
gdown.download(url, output, quiet=False)

df = pd.read_csv(output)
print("Dataset Loaded:")
print(df.head())
print(df.shape)

# ---------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------
df["balanceDiffOrig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
df["balanceDiffDest"] = df["newbalanceDest"] - df["oldbalanceDest"]

model_cols = [
    "type","amount",
    "oldbalanceOrg","newbalanceOrig",
    "oldbalanceDest","newbalanceDest",
    "balanceDiffOrig","balanceDiffDest",
    "isFraud"
]

df_model = df[model_cols]

X = df_model.drop("isFraud", axis=1)
y = df_model["isFraud"]

categorical = ["type"]
numeric = [
    "amount","oldbalanceOrg","newbalanceOrig",
    "oldbalanceDest","newbalanceDest",
    "balanceDiffOrig","balanceDiffDest"
]

# ---------------------------------------------------------
# 3. TRAIN/TEST SPLIT
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ---------------------------------------------------------
# 4. PREPROCESSOR (Scaler + Encoder)
# ---------------------------------------------------------
preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(drop="first"), categorical),
    ]
)

# ---------------------------------------------------------
# 5. LOGISTIC REGRESSION
# ---------------------------------------------------------
log_reg = Pipeline([
    ("prep", preprocess),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=2000))
])

print("\nTraining Logistic Regression...")
log_reg.fit(X_train, y_train)
print(classification_report(y_test, log_reg.predict(X_test)))

joblib.dump(log_reg, "logistic_regression.pkl")

# ---------------------------------------------------------
# 6. RANDOM FOREST
# ---------------------------------------------------------
rf = Pipeline([
    ("prep", preprocess),
    ("clf", RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42
    ))
])

print("\nTraining Random Forest...")
rf.fit(X_train, y_train)
print(classification_report(y_test, rf.predict(X_test)))

joblib.dump(rf, "random_forest.pkl")

# ---------------------------------------------------------
# 7. XGBOOST
# ---------------------------------------------------------
xgb = Pipeline([
    ("prep", preprocess),
    ("clf", XGBClassifier(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        scale_pos_weight=10  # IMPORTANT FOR IMBALANCED DATA
    ))
])

print("\nTraining XGBoost...")
xgb.fit(X_train, y_train)
print(classification_report(y_test, xgb.predict(X_test)))

joblib.dump(xgb, "xgboost.pkl")

# ---------------------------------------------------------
# 8. ISOLATION FOREST (UNSUPERVISED)
# ---------------------------------------------------------
print("\nTraining Isolation Forest...")
iso_forest = IsolationForest(
    n_estimators=200,
    contamination=0.002,
    random_state=42
)

# Fit ONLY on numeric features
iso_forest.fit(X_train[numeric])

joblib.dump(iso_forest, "isolation_forest.pkl")

# ---------------------------------------------------------
# 9. HYBRID MODEL (ENSEMBLE)
# ---------------------------------------------------------
class HybridModel:
    def __init__(self, model1, model2, model3):
        self.m1 = model1
        self.m2 = model2
        self.m3 = model3

    def predict(self, X):
        p1 = self.m1.predict(X)
        p2 = self.m2.predict(X)
        p3 = self.m3.predict(X[numeric])
        p3 = np.where(p3 == -1, 1, 0)

        # majority vote
        predictions = (p1 + p2 + p3 >= 2).astype(int)
        return predictions

hybrid = HybridModel(log_reg, rf, iso_forest)

# Save Hybrid model
joblib.dump(hybrid, "hybrid_model.pkl")

print("\nAll model training COMPLETE!")
print("Saved Files:")
print("""
- logistic_regression.pkl
- random_forest.pkl
- xgboost.pkl
- isolation_forest.pkl
- hybrid_model.pkl
""")

