import os
import time
import pickle
import torch
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from transformers import AutoTokenizer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_curve, auc, precision_recall_curve, log_loss, matthews_corrcoef, 
    cohen_kappa_score, brier_score_loss, balanced_accuracy_score
)

# Load variables from .env file
load_dotenv()

mlflowURI = os.getenv("MLFLOW_TRACKING_URI")
weight_path = os.getenv("MODEL_WIGHTS_PATH")

print("MLflow URI: ", mlflowURI)
print("Model Weights Path: ", weight_path)

#sleep for 60 seconds
# time.sleep(60)

# MLflow Setup
mlflow.set_tracking_uri(mlflowURI)  # Connect to MLflow server
mlflow.set_experiment("Text_Classification_Evaluation")  # Create/Use experiment

# Load Model
with open(weight_path, "rb") as f:
    model = pickle.load(f)

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Ensure Model is in Eval Mode
model.eval()

# Prediction Function (returns probability of positive class)
def predict_proba(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=1).squeeze().tolist()
    return probs[1]  # Probability of class 1 (Positive)

# Sample Test Dataset
test_data = [
    ("I love this product, it's amazing!", 1),
    ("This is the worst thing I have ever bought.", 0),
    ("Absolutely fantastic! Highly recommend.", 1),
    ("Not worth the money, very disappointing.", 0),
    ("Could be better, but overall decent.", 1),
    ("Terrible experience, I regret buying it.", 0),
    ("Really good quality, I'm happy with it.", 1),
    ("Awful product, do not buy!", 0)
]

# Get Predictions
true_labels = [label for _, label in test_data]
predicted_probs = [predict_proba(text) for text, _ in test_data]

# Convert Probabilities to Binary Predictions (Threshold = 0.5)
threshold = 0.5
predicted_labels = [1 if prob >= threshold else 0 for prob in predicted_probs]

# Compute Confusion Matrix Elements
TP = sum((p == 1 and t == 1) for p, t in zip(predicted_labels, true_labels))
TN = sum((p == 0 and t == 0) for p, t in zip(predicted_labels, true_labels))
FP = sum((p == 1 and t == 0) for p, t in zip(predicted_labels, true_labels))
FN = sum((p == 0 and t == 1) for p, t in zip(predicted_labels, true_labels))

# Compute Metrics
metrics = {
    "accuracy": accuracy_score(true_labels, predicted_labels),
    "precision": precision_score(true_labels, predicted_labels, zero_division=1),
    "recall": recall_score(true_labels, predicted_labels),
    "specificity": TN / (TN + FP) if (TN + FP) > 0 else 0,
    "f1_score": f1_score(true_labels, predicted_labels),
    "youden_index": recall_score(true_labels, predicted_labels) + (TN / (TN + FP)) - 1,
    "mcc": matthews_corrcoef(true_labels, predicted_labels),
    "auc_roc": auc(*roc_curve(true_labels, predicted_probs)[:2]),
    "logloss": log_loss(true_labels, predicted_probs),
    "brier_score": brier_score_loss(true_labels, predicted_probs),
    "cohen_kappa": cohen_kappa_score(true_labels, predicted_labels),
    "balanced_accuracy": balanced_accuracy_score(true_labels, predicted_labels)
}

# Generate & Save ROC Curve
fpr, tpr, _ = roc_curve(true_labels, predicted_probs)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"AUC-ROC = {metrics['auc_roc']:.2f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC Curve")
plt.legend()
plt.savefig("roc_curve.png")  # Save ROC Curve

# Generate & Save Precision-Recall Curve
precision_vals, recall_vals, _ = precision_recall_curve(true_labels, predicted_probs)
plt.figure(figsize=(7, 5))
plt.plot(recall_vals, precision_vals, label="Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.savefig("precision_recall_curve.png")  # Save PR Curve

# Log Metrics & Artifacts to MLflow
with mlflow.start_run():
    mlflow.log_params({"threshold": threshold})  # Log threshold used
    mlflow.log_metrics(metrics)  # Log all classification metrics
    mlflow.log_artifact("roc_curve.png")  # Log ROC curve image
    mlflow.log_artifact("precision_recall_curve.png")  # Log Precision-Recall curve image
    mlflow.sklearn.log_model(model, "text_classification_model")  # Log model

print("✅ Metrics logged to MLflow at ", mlflowURI)

# Keep the process active
# print("Process is now active. Press Ctrl+C to terminate.")
# while True:
#     pass
