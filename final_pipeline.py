import pandas as pd
import numpy as np
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# import raw data and best model name from train_models.py environment
from train_models import X, y, best_model_name

# Ensure X is a dataframe (for column transformer compatibility)
if isinstance(X, np.ndarray):
    X = pd.DataFrame(X, columns=[
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
        'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ])

print("\n========================================")
print(" best model selected:", best_model_name.upper())
print("========================================\n")

# Load the full fitted pipeline (preprocessor + model) saved by train_models.py
model_path = f"models/{best_model_name}.joblib"
print(f"loading model pipeline from: {model_path}")
final_pipeline = joblib.load(model_path)
print("pipeline loaded successfully.\n")

# Train/test split (reproducible)
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("data split completed.\n")

# Retrain the entire pipeline on the training data
print("training final pipeline...\n")
final_pipeline.fit(x_train, y_train)
print("training completed.\n")

# Predict and evaluate on test set
y_pred = final_pipeline.predict(x_test)

# Some classifiers may have predict_proba; fallback to decision_function if needed
if hasattr(final_pipeline, "predict_proba"):
    y_prob = final_pipeline.predict_proba(x_test)[:, 1]
elif hasattr(final_pipeline, "decision_function"):
    y_prob = final_pipeline.decision_function(x_test)
else:
    y_prob = None

metrics = {
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "precision": float(precision_score(y_test, y_pred)),
    "recall": float(recall_score(y_test, y_pred)),
    "f1_score": float(f1_score(y_test, y_pred)),
    "roc_auc": float(roc_auc_score(y_test, y_prob)) if y_prob is not None else None
}

print("final model metrics:", metrics, "\n")

# Save metrics to JSON
with open("reports/final_pipeline_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)
print("saved metrics to reports/final_pipeline_metrics.json\n")

# Save retrained pipeline again
joblib.dump(final_pipeline, "models/final_pipeline.joblib")
print("final pipeline saved as models/final_pipeline.joblib\n")

print("========================================")
print(" final pipeline ready")
print("========================================")
