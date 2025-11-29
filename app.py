# --------------------------------------------
# 1. IMPORT LIBRARIES
# --------------------------------------------
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
sns.set(style="whitegrid")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

# --------------------------------------------
# 2. LOAD DATASET FROM GOOGLE DRIVE
# --------------------------------------------
# Replace with your Google Drive File ID
FILE_ID = "1Kwm31irBeeMqRmyp6fumkGSxlnGSmrzY"  
CSV_URL = f"https://drive.google.com/uc?id={FILE_ID}"

df = pd.read_csv(CSV_URL)
df.head()

# Ensure 'isFraud' column exists
if "isFraud" not in df.columns:
    raise ValueError("Dataset does not have 'isFraud' column.")

# --------------------------------------------
# 3. BASIC ANALYSIS
# --------------------------------------------
print(df.shape)
print(df.isnull().sum())
print(df["isFraud"].value_counts())
print("Fraud %:", round(df["isFraud"].mean()*100,2))

# --------------------------------------------
# 4. FEATURE ENGINEERING
# --------------------------------------------
df["balanceDiffOrig"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
df["balanceDiffDest"] = df["newbalanceDest"] - df["oldbalanceDest"]

df["zero_after_transfer"] = ((df["oldbalanceOrg"] > 0) & 
                             (df["newbalanceOrig"] == 0) & 
                             (df["type"].isin(["TRANSFER","CASH_OUT"]))).astype(int)

# --------------------------------------------
# 5. VISUALIZATIONS
# --------------------------------------------
sns.countplot(df["isFraud"])
plt.title("Fraud vs Non-Fraud")
plt.show()

sns.countplot(x="type", hue="isFraud", data=df)
plt.title("Transaction Type vs Fraud")
plt.show()

numeric_cols = ["amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest","balanceDiffOrig","balanceDiffDest","zero_after_transfer","isFraud"]
sns.heatmap(df[numeric_cols].corr(), cmap="coolwarm", annot=True)
plt.title("Correlation Heatmap")
plt.show()

# --------------------------------------------
# 6. PREPARE DATA FOR ML
# --------------------------------------------
features = ["amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest","balanceDiffOrig","balanceDiffDest","zero_after_transfer","type"]
X = df[features]
y = df["isFraud"]

categorical = ["type"]
numeric = ["amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest","balanceDiffOrig","balanceDiffDest","zero_after_transfer"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(drop="first"), categorical)
    ]
)

# --------------------------------------------
# 7. TRAIN MODELS AND SAVE
# --------------------------------------------
# Logistic Regression
lr_pipeline = Pipeline([
    ("prep", preprocessor),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=2000))
])
lr_pipeline.fit(X_train, y_train)
joblib.dump(lr_pipeline,"logistic_regression.pkl")

# Random Forest
rf_pipeline = Pipeline([
    ("prep", preprocessor),
    ("clf", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42))
])
rf_pipeline.fit(X_train, y_train)
joblib.dump(rf_pipeline,"random_forest.pkl")

# XGBoost
xgb_pipeline = Pipeline([
    ("prep", preprocessor),
    ("clf", XGBClassifier(scale_pos_weight=int(y_train.value_counts()[0]/y_train.value_counts()[1]), use_label_encoder=False, eval_metric="logloss"))
])
xgb_pipeline.fit(X_train, y_train)
joblib.dump(xgb_pipeline,"xgboost.pkl")

# Isolation Forest (unsupervised)
iso = IsolationForest(contamination=y_train.mean(), random_state=42)
iso.fit(X_train[numeric])
joblib.dump(iso,"isolation_forest.pkl")

# Hybrid Model
hybrid_models = {"rf": rf_pipeline, "xgb": xgb_pipeline, "iso": iso}
joblib.dump(hybrid_models,"hybrid_model.pkl")

# --------------------------------------------
# 8. EVALUATE MODELS
# --------------------------------------------
def evaluate_model(model, X_test, y_test, model_name):
    if model_name != "Isolation Forest":
        y_pred = model.predict(X_test)
        print(f"--- {model_name} ---")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("ROC-AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))
    else:
        y_pred = model.predict(X_test[numeric])
        y_pred = np.where(y_pred==-1,1,0)
        print(f"--- {model_name} ---")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

evaluate_model(lr_pipeline, X_test, y_test, "Logistic Regression")
evaluate_model(rf_pipeline, X_test, y_test, "Random Forest")
evaluate_model(xgb_pipeline, X_test, y_test, "XGBoost")
evaluate_model(iso, X_test, y_test, "Isolation Forest")

print("All models trained and saved as .pkl files")

