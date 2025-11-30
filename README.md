# 💳 Credit Card Fraud Detection Dashboard

This is an interactive **Streamlit web app** for detecting fraudulent credit card transactions using a trained machine learning model. Users can input transaction details and get real-time fraud predictions with probability scores.

---

## 📌 Features

* Predict fraud for single credit card transactions.
* Display transaction summary and computed balance differences.
* Interactive **fraud probability gauge**.
* Optional visualizations: transaction type distribution and balance differences.
* Model hosted on **Google Drive** for lightweight deployment.

---

## 🛠️ Technology Stack

* **Machine Learning:** Logistic Regression, Random Forest, XGBoost (Hybrid Model)
* **Data Processing:** Pandas, NumPy
* **Visualization:** Plotly, Streamlit
* **Deployment:** Streamlit Cloud

---

## 🚀 How to Run

1. Clone the repository:

```bash
git clone <your-repo-link>
cd credit-card-fraud-detection
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the Streamlit app:

```bash
streamlit run app.py
```

4. Open the app in your browser at `http://localhost:8501`.

---

## 📝 Transaction Input

Users can input:

* Transaction Type (e.g., PAYMENT, TRANSFER, CASH_OUT)
* Transaction Amount
* Sender Old/New Balance
* Receiver Old/New Balance

The app calculates balance differences and predicts fraud probability.

---

## 📦 Folder Structure

```
credit-card-fraud-detection/
├─ app.py
├─ requirements.txt
├─ README.md
└─ data/ (optional, if dataset needed)
```

---

## 🔗 Model Hosting

The model (`fraud_detection_pipeline.pkl`) is hosted on Google Drive for easy access:

* Google Drive link: [Model File](https://drive.google.com/file/d/1Hjxc5wS13dMRWJkNUhRLEXRPBo5tga0Z/view?usp=share_link)

---

## ⚡ Notes

* Ensure the Google Drive file permissions are set to "Anyone with the link can view".
* The app works with **single transaction inputs**. Future updates may include batch predictions via CSV upload.

---

## 📧 Contact

Developed by: **Gokul G**
For questions or issues, open an issue in the GitHub repository.
