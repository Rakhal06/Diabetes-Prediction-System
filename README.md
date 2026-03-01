# 🚀 Diabetes Risk Prediction System

## 🌐 Live Demo  
👉 **Deployed App:**  
https://diabetes-prediction-system-amwusbsaqrcspfgv6zskka.streamlit.app  

---

## 📌 Project Overview

The **Diabetes Risk Prediction System** is an end-to-end Machine Learning application that predicts the likelihood of diabetes using patient medical attributes.

This project demonstrates:

- Data preprocessing & cleaning  
- Feature scaling  
- Model comparison  
- Hyperparameter tuning  
- Model evaluation (ROC, AUC, Confusion Matrix)  
- Model deployment using Streamlit Cloud  

This is a production-style ML workflow, not just a notebook experiment.

---

## 🧠 Dataset

- **Pima Indians Diabetes Dataset**
- 768 samples
- 8 medical predictor features
- Binary classification:
  - `0` → No Diabetes
  - `1` → Diabetes

---

## 🔎 Models Implemented

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| Logistic Regression | 71% | 0.66 |
| Random Forest (Tuned) | **75%** | **0.74** |

Random Forest was selected as the final deployed model after hyperparameter tuning.

---

## ⚙️ Machine Learning Pipeline

1. Data Cleaning (zero-value replacement with median)
2. Feature Scaling using StandardScaler
3. Train-Test Split
4. Baseline Model: Logistic Regression
5. Random Forest with:
   - `class_weight='balanced'`
   - GridSearchCV for tuning
6. Cross-validation
7. Model Evaluation:
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - ROC-AUC
8. Model Saving using `joblib`
9. Deployment using Streamlit Cloud

---

## 📈 Model Evaluation

Final Random Forest Performance:

- **Accuracy:** 74.6%
- **ROC-AUC:** 0.745
- Balanced performance on imbalanced dataset
- Confusion Matrix analysis
- False Positive & False Negative evaluation

The deployed application also includes:
- ROC Curve visualization
- Feature Importance chart
- Probability confidence scores

---

## 💻 Web Application Features

- Interactive patient input form
- Model selection (Logistic Regression / Random Forest)
- Risk prediction output (Low / High Risk)
- Confidence probability display
- ROC Curve visualization
- Feature importance chart
- Medical disclaimer
- Clean production UI (Dark theme)

---

## 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit
- Git & GitHub
- Streamlit Cloud

---

## 📂 Project Structure
Diabetes-Prediction-System/
│
├── Data/
│ └── pima-indians-diabetes.csv
│
├── app.py
├── Train_Model.py
├── diabetes_model.joblib
├── logistic_model.joblib
├── scaler.joblib
├── requirements.txt
└── README.md

⚠ Medical Disclaimer

This application is for educational purposes only and should not be considered a medical diagnosis. Always consult a healthcare professional.

🎯 Key Learning Outcomes

Handling imbalanced datasets

Hyperparameter tuning using GridSearchCV

Evaluating ML models beyond accuracy

Saving & loading production models

Deploying ML applications to the cloud

Building interactive ML dashboards

👨‍💻 Author

Rakhal Krishna
AI & Data Science Student

GitHub: https://github.com/Rakhal06


---

## 🚀 Run Locally

```bash
git clone https://github.com/Rakhal06/Diabetes-Prediction-System.git
cd Diabetes-Prediction-System
python -m venv venv
venv\Scripts\activate   # Windows
pip install -r requirements.txt
streamlit run app.py

