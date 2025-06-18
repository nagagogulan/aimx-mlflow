import os
import pandas as pd
import numpy as np
import torch
import mlflow
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score, log_loss, cohen_kappa_score
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import LabelEncoder

# Load variables from .env file
load_dotenv()

mlflowURI = os.getenv("MLFLOW_TRACKING_URI")
weight_path = os.getenv("MODEL_WIGHTS_PATH")
dataset_path = os.getenv("DATASET_PATH")
target_column = os.getenv("TARGET_COLUMN")
experiment_name = os.getenv("EXPERIMENT_NAME")

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 1: Load the dataset
df = pd.read_csv(dataset_path)

# Step 2: Separate features and target
X = df.drop(columns=[target_column])
y = df[target_column]

# Step 3: Preprocess if needed
# Convert categorical features to numeric
for col in X.select_dtypes(include=['object']).columns:
    X[col] = pd.factorize(X[col])[0]

# Encode target labels for classification
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y_encoded, dtype=torch.long).to(device)

# Step 4: Load your trained PyTorch model
model = torch.load(weight_path, map_location=device)
model.eval()  # Set the model to evaluation mode

# Step 5: Predict
with torch.no_grad():
    # Get raw predictions (logits)
    logits = model(X_tensor)
    
    # Convert logits to probabilities
    if num_classes > 2:
        # Multi-class: use softmax
        y_pred_proba = torch.softmax(logits, dim=1).cpu().numpy()
        # Get class with highest probability
        y_pred = np.argmax(y_pred_proba, axis=1)
    else:
        # Binary classification
        if logits.shape[1] == 1:
            # Single output neuron with sigmoid
            y_pred_proba = torch.sigmoid(logits).cpu().numpy().flatten()
            y_pred = (y_pred_proba > 0.5).astype(int)
            # For ROC-AUC calculation, reshape to match sklearn's format
            y_prob = np.column_stack([1-y_pred_proba, y_pred_proba])
        else:
            # Two output neurons with softmax
            y_pred_proba = torch.softmax(logits, dim=1).cpu().numpy()
            y_pred = np.argmax(y_pred_proba, axis=1)
            y_prob = y_pred_proba

# Convert predictions back to original labels
predictions = label_encoder.inverse_transform(y_pred)

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
    if num_classes == 2:
        # Binary classification
        if logits.shape[1] == 1:
            # Single output neuron
            auc_roc = roc_auc_score(y_encoded, y_pred_proba)
        else:
            # Two output neurons
            auc_roc = roc_auc_score(y_encoded, y_pred_proba[:, 1])
    else:
        # Multi-class: use one-vs-rest approach (ovr)
        y_one_hot = np.zeros((len(y_encoded), num_classes))
        for i in range(len(y_encoded)):
            y_one_hot[i, y_encoded[i]] = 1
        auc_roc = roc_auc_score(y_one_hot, y_pred_proba, multi_class='ovr')
except Exception as e:
    print(f"Warning: Could not calculate AUC-ROC: {e}")

# Calculate Logarithmic Loss (LogLoss)
logloss = 0  # Default value
try:
    if num_classes == 2:
        # Binary classification
        if logits.shape[1] == 1:
            # Single output neuron
            logloss = log_loss(y_encoded, y_pred_proba)
        else:
            # Two output neurons
            logloss = log_loss(y_encoded, y_pred_proba)
    else:
        # Multi-class
        y_one_hot = np.zeros((len(y_encoded), num_classes))
        for i in range(len(y_encoded)):
            y_one_hot[i, y_encoded[i]] = 1
        logloss = log_loss(y_one_hot, y_pred_proba)
except Exception as e:
    print(f"Warning: Could not calculate LogLoss: {e}")

# Calculate Brier Score: (1/N) Σ (y_i - ŷ_i)^2
brier_score = 0  # Default value
try:
    if num_classes == 2:
        # Binary classification
        if logits.shape[1] == 1:
            # Single output neuron
            brier_score = brier_score_loss(y_encoded, y_pred_proba)
        else:
            # Two output neurons
            brier_score = brier_score_loss(y_encoded, y_pred_proba[:, 1])
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

# PyTorch-specific metrics
pytorch_metrics = {}

# Get model architecture
pytorch_metrics["model_architecture"] = str(model)

# Count number of layers
layer_count = 0
for _ in model.modules():
    layer_count += 1
pytorch_metrics["num_layers"] = layer_count - 1  # Subtract 1 to exclude the model itself

# Count trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
pytorch_metrics["trainable_params"] = trainable_params

# Count total parameters
total_params = sum(p.numel() for p in model.parameters())
pytorch_metrics["total_params"] = total_params

# Get model size in MB
model_size_bytes = 0
for param in model.parameters():
    model_size_bytes += param.nelement() * param.element_size()
pytorch_metrics["model_size_mb"] = model_size_bytes / (1024 * 1024)

# Feature importance (for interpretable models)
feature_importance = {}
# For simple models, we can use the weights of the first linear layer as a proxy for feature importance
try:
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Get the weights of the first linear layer
            weights = module.weight.data.cpu().numpy()
            # Calculate importance as the mean absolute weight for each feature
            importances = np.mean(np.abs(weights), axis=0)
            for i, importance in enumerate(importances):
                if i < len(X.columns):
                    feature_importance[X.columns[i]] = float(importance)
            break
except Exception as e:
    print(f"Warning: Could not calculate feature importance: {e}")

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

# Print PyTorch-specific metrics
print("\nPyTorch-specific metrics:")
for metric_name, value in pytorch_metrics.items():
    if isinstance(value, (int, float)):
        print(f"{metric_name}: {value:.4f}")
    elif metric_name != "model_architecture":  # Don't print the full model architecture
        print(f"{metric_name}: {value}")
print(f"Number of layers: {pytorch_metrics['num_layers']}")
print(f"Trainable parameters: {pytorch_metrics['trainable_params']}")
print(f"Model size (MB): {pytorch_metrics['model_size_mb']:.2f}")

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
    
    # Log PyTorch-specific metrics
    for metric_name, value in pytorch_metrics.items():
        if isinstance(value, (int, float)):
            mlflow.log_metric(metric_name, value)
        elif metric_name == "model_architecture":
            # Log model architecture as a text artifact
            with open("model_architecture.txt", "w") as f:
                f.write(value)
            mlflow.log_artifact("model_architecture.txt")
        else:
            mlflow.log_param(metric_name, value)
    
    # You can log additional information if needed
    mlflow.log_param("model_path", weight_path)
    mlflow.log_param("dataset_path", dataset_path)
    mlflow.log_param("num_classes", num_classes)
    mlflow.log_param("device", str(device))
    
    # Log the PyTorch model
    try:
        mlflow.pytorch.log_model(model, "model")
    except:
        print("Warning: Could not log PyTorch model to MLflow")
    
    print(f"Metrics successfully logged to MLflow experiment: {experiment_name}")