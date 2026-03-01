import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    precision_recall_curve
)

# -------------------------------------
# Step 1: Load Dataset
# -------------------------------------
print("Loading dataset...")
columns = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
]

data = pd.read_csv("Data/pima-indians-diabetes.csv", header=None, names=columns)
print("Dataset shape:", data.shape)

# -------------------------------------
# Step 2: Data Cleaning
# -------------------------------------
print("\nCleaning data...")

columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
data[columns_with_zeros] = data[columns_with_zeros].replace(0, np.nan)

for col in columns_with_zeros:
    data[col].fillna(data[col].median(), inplace=True)

print("Zero values replaced with median.")

# -------------------------------------
# Step 3: Train-Test Split
# -------------------------------------
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------------
# Step 4: Scaling
# -------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------------
# Step 5: Baseline Logistic Regression
# -------------------------------------
print("\nBaseline Model: Logistic Regression")
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)
lr_preds = lr.predict(X_test_scaled)

print(classification_report(y_test, lr_preds))
print("ROC-AUC:", roc_auc_score(y_test, lr_preds))

# -------------------------------------
# Step 6: Random Forest with Class Weight
# -------------------------------------
print("\nTraining Random Forest with class_weight='balanced'")

rf = RandomForestClassifier(
    random_state=42,
    class_weight='balanced'
)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid = GridSearchCV(
    rf,
    param_grid,
    cv=5,
    scoring='recall',   # Important for medical prediction
    n_jobs=-1
)

grid.fit(X_train_scaled, y_train)

best_model = grid.best_estimator_

print("\nBest Parameters Found:")
print(grid.best_params_)

# -------------------------------------
# Step 7: Cross Validation
# -------------------------------------
cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5)
print("\nCross Validation Accuracy:", cv_scores.mean())

# -------------------------------------
# Step 8: Evaluation
# -------------------------------------
print("\nFinal Random Forest Evaluation")

rf_preds = best_model.predict(X_test_scaled)

print(classification_report(y_test, rf_preds))
print("Accuracy:", accuracy_score(y_test, rf_preds))
print("ROC-AUC:", roc_auc_score(y_test, rf_preds))

cm = confusion_matrix(y_test, rf_preds)
tn, fp, fn, tp = cm.ravel()

print("\nConfusion Matrix:")
print(cm)
print("False Negatives:", fn)
print("False Positives:", fp)

# -------------------------------------
# Step 9: Precision-Recall Curve
# -------------------------------------
probs = best_model.predict_proba(X_test_scaled)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, probs)

plt.figure(figsize=(6, 4))
plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show()

# -------------------------------------
# Step 10: Feature Importance
# -------------------------------------
importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": best_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importance:")
print(importance_df)

plt.figure(figsize=(8, 5))
sns.barplot(x="Importance", y="Feature", data=importance_df)
plt.title("Feature Importance")
plt.show()



# -------------------------------------
# Step 11: Save Final Model & Scaler
# -------------------------------------
print("\nSaving best model and scaler...")

X_scaled_full = scaler.fit_transform(X)
best_model.fit(X_scaled_full, y)

# Save Logistic Regression too
lr.fit(X_scaled_full, y)
joblib.dump(lr, "logistic_model.joblib")

joblib.dump(best_model, "diabetes_model.joblib")
joblib.dump(scaler, "scaler.joblib")

print("Model and scaler saved successfully.")

print("\n--- PIPELINE COMPLETE ---")