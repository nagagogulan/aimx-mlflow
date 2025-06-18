import os
import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_decision_forests as tfdf
import mlflow
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score, roc_curve, log_loss, cohen_kappa_score
from sklearn.metrics import brier_score_loss

# Load variables from .env file
load_dotenv()

mlflowURI = os.getenv("MLFLOW_TRACKING_URI")
weight_path = os.getenv("MODEL_WIGHTS_PATH")
dataset_path = os.getenv("DATASET_PATH")
target_column = os.getenv("TARGET_COLUMN")
experiment_name = os.getenv("EXPERIMENT_NAME")

# Step 1: Load the dataset
df = pd.read_csv(dataset_path)

# Step 2: Separate features and target
X = df.drop(columns=[target_column])
y = df[target_column]

# Step 3: Preprocess data for TensorFlow
# Convert categorical features to string (TF Decision Forests handles categorical features automatically)
for col in X.select_dtypes(include=['object', 'category']).columns:
    X[col] = X[col].astype(str)

# Create TensorFlow dataset
def df_to_dataset(features, labels, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.batch(batch_size)
    return dataset

batch_size = 32
tf_dataset = df_to_dataset(X, y, batch_size)

# Step 4: Load your trained model
model = tf.keras.models.load_model(weight_path)

# Step 5: Predict
predictions_prob = model.predict(tf_dataset)
predictions = np.argmax(predictions_prob, axis=1) if len(predictions_prob.shape) > 1 else (predictions_prob > 0.5).astype(int)

# Map numeric predictions back to original class labels if needed
unique_classes = sorted(y.unique())
if len(unique_classes) > 2:
    # For multi-class
    predictions = [unique_classes[pred] for pred in predictions]
else:
    # For binary classification
    predictions = [unique_classes[1] if pred == 1 else unique_classes[0] for pred in predictions]

# Step 6: Compare with actuals
results = pd.DataFrame({
    "Actual": y,
    "Predicted": predictions
})

print(results)

# Step 7: Calculate Evaluation Metrics
# Count correct predictions (TP + TN)
correct_predictions = (results["Actual"] == results["Predicted"]).sum()
# Total number of predictions (TP + TN + FP + FN)
total_predictions = len(results)
# Calculate accuracy using the formula (TP + TN) / (TP + TN + FP + FN)
accuracy = correct_predictions / total_predictions

# Define the positive class
positive_class = "TruePositive"

# True Positives: predicted positive and actually positive
true_positives = ((results["Predicted"] == positive_class) & 
                 (results["Actual"] == positive_class)).sum()

# False Positives: predicted positive but actually negative
false_positives = ((results["Predicted"] == positive_class) & 
                  (results["Actual"] != positive_class)).sum()

# False Negatives: predicted negative but actually positive
false_negatives = ((results["Predicted"] != positive_class) & 
                  (results["Actual"] == positive_class)).sum()

# Calculate precision using the formula: TP / (TP + FP)
precision = 0  # Default value
if (true_positives + false_positives) > 0:  # Avoid division by zero
    precision = true_positives / (true_positives + false_positives)

# Calculate recall using the formula: TP / (TP + FN)
recall = 0  # Default value
if (true_positives + false_negatives) > 0:  # Avoid division by zero
    recall = true_positives / (true_positives + false_negatives)

# Calculate specificity (TNR) using the formula: TN / (TN + FP)
# True Negatives: predicted negative and actually negative
true_negatives = ((results["Predicted"] != positive_class) & 
                 (results["Actual"] != positive_class)).sum()
                 
specificity = 0  # Default value
if (true_negatives + false_positives) > 0:  # Avoid division by zero
    specificity = true_negatives / (true_negatives + false_positives)

# Calculate F1-score using the formula: 2 * (Precision * Recall) / (Precision + Recall)
f1_score = 0  # Default value
if (precision + recall) > 0:  # Avoid division by zero
    f1_score = 2 * (precision * recall) / (precision + recall)

# Calculate Youden's Index (J-statistic) using the formula: Sensitivity + Specificity - 1
# Note: Sensitivity is the same as Recall
youdens_index = recall + specificity - 1

# Calculate Matthews Correlation Coefficient (MCC)
mcc_numerator = (true_positives * true_negatives) - (false_positives * false_negatives)
mcc_denominator = ((true_positives + false_positives) * 
                   (true_positives + false_negatives) * 
                   (true_negatives + false_positives) * 
                   (true_negatives + false_negatives))

# Default value
mcc = 0
# Avoid division by zero and square root of negative number
if mcc_denominator > 0:
    mcc = mcc_numerator / (mcc_denominator ** 0.5)

# Calculate AUC-ROC (Area Under the Receiver Operating Characteristic Curve)
auc_roc = 0  # Default value
try:
    # For binary classification
    if len(unique_classes) == 2:
        # Binary classification: use probabilities of the positive class
        y_binary = (y == unique_classes[1]).astype(int)
        if len(predictions_prob.shape) > 1:
            auc_roc = roc_auc_score(y_binary, predictions_prob[:, 1])
        else:
            auc_roc = roc_auc_score(y_binary, predictions_prob)
    else:
        # Multi-class: use one-vs-rest approach (ovr)
        y_one_hot = pd.get_dummies(y)
        auc_roc = roc_auc_score(y_one_hot, predictions_prob, multi_class='ovr')
except Exception as e:
    print(f"Warning: Could not calculate AUC-ROC: {e}")

# Calculate Logarithmic Loss (LogLoss)
logloss = 0  # Default value
try:
    # For binary classification
    if len(unique_classes) == 2:
        y_binary = (y == unique_classes[1]).astype(int)
        if len(predictions_prob.shape) > 1:
            logloss = log_loss(y_binary, predictions_prob[:, 1])
        else:
            logloss = log_loss(y_binary, predictions_prob)
    else:
        # Multi-class
        y_one_hot = pd.get_dummies(y)
        logloss = log_loss(y_one_hot, predictions_prob)
except Exception as e:
    print(f"Warning: Could not calculate LogLoss: {e}")

# Calculate Brier Score: (1/N) Σ (y_i - ŷ_i)^2
brier_score = 0  # Default value
try:
    if len(unique_classes) == 2:
        y_binary = (y == unique_classes[1]).astype(int)
        if len(predictions_prob.shape) > 1:
            brier_score = brier_score_loss(y_binary, predictions_prob[:, 1])
        else:
            brier_score = brier_score_loss(y_binary, predictions_prob)
except Exception as e:
    print(f"Warning: Could not calculate Brier Score: {e}")

# Calculate Cohen's Kappa (inter-annotator agreement)
cohens_kappa = 0  # Default value
try:
    # Calculate Cohen's kappa using scikit-learn's cohen_kappa_score function
    cohens_kappa = cohen_kappa_score(y, predictions)
except Exception as e:
    print(f"Warning: Could not calculate Cohen's Kappa: {e}")

# Calculate Balanced Accuracy: (Sensitivity + Specificity) / 2
# Note: Sensitivity is the same as Recall
balanced_accuracy = (recall + specificity) / 2

# TensorFlow Decision Forests specific metrics
# Feature importance
feature_importance = {}
try:
    if isinstance(model, tfdf.keras.RandomForestModel) or isinstance(model, tfdf.keras.GradientBoostedTreesModel):
        # Get feature importance from TF Decision Forests model
        importance_dict = model.make_inspector().variable_importances()
        if "MEAN_DECREASE_IN_ACCURACY" in importance_dict:
            importances = importance_dict["MEAN_DECREASE_IN_ACCURACY"]
            for feature_name, importance in importances:
                feature_importance[feature_name] = float(importance)
except Exception as e:
    print(f"Warning: Could not extract feature importance: {e}")

print(f"\nAccuracy: {accuracy:.4f} ({correct_predictions}/{total_predictions})")
print(f"Precision: {precision:.4f} ({true_positives}/{true_positives + false_positives})")
print(f"Recall: {recall:.4f} ({true_positives}/{true_positives + false_negatives})")
print(f"Specificity: {specificity:.4f} ({true_negatives}/{true_negatives + false_positives})")
print(f"F1-score: {f1_score:.4f}")
print(f"Youden's Index: {youdens_index:.4f}")
print(f"Matthews Correlation Coefficient: {mcc:.4f}")
print(f"AUC-ROC: {auc_roc:.4f}")
print(f"LogLoss: {logloss:.4f}")
print(f"Brier Score: {brier_score:.4f}")
print(f"Cohen's Kappa: {cohens_kappa:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy:.4f}")

# Step 8: Log metrics to MLflow
# Set the MLflow tracking URI
mlflow.set_tracking_uri(mlflowURI)

# Set the experiment
mlflow.set_experiment(experiment_name)

# Start an MLflow run
with mlflow.start_run():
    # Log the metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("specificity", specificity)
    mlflow.log_metric("f1_score", f1_score)
    mlflow.log_metric("youdens_index", youdens_index)
    mlflow.log_metric("mcc", mcc)
    mlflow.log_metric("auc_roc", auc_roc)
    mlflow.log_metric("logloss", logloss)
    mlflow.log_metric("brier_score", brier_score)
    mlflow.log_metric("cohens_kappa", cohens_kappa)
    mlflow.log_metric("balanced_accuracy", balanced_accuracy)
    
    # Log feature importance
    for feature, importance in feature_importance.items():
        mlflow.log_metric(f"importance_{feature}", importance)
    
    # You can log additional information if needed
    mlflow.log_param("model_path", weight_path)
    mlflow.log_param("dataset_path", dataset_path)
    
    print(f"Metrics successfully logged to MLflow experiment: {experiment_name}")