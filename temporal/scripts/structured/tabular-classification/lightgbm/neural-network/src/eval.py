import os
import pandas as pd
import numpy as np
import lightgbm as lgb
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

# Step 4: Load your trained LightGBM model
model = lgb.Booster(model_file=weight_path)

# Step 5: Predict
# Get raw predictions (margin values)
if num_classes > 2:
    # Multi-class classification
    y_pred_proba = model.predict(X)
    # Reshape if needed
    if len(y_pred_proba.shape) == 1:
        y_pred_proba = y_pred_proba.reshape(X.shape[0], num_classes)
    # Get class with highest probability
    y_pred = np.argmax(y_pred_proba, axis=1)
else:
    # Binary classification
    y_pred_proba = model.predict(X)
    # Convert to binary predictions
    y_pred = (y_pred_proba > 0.5).astype(int)
    # For ROC-AUC calculation, reshape to match sklearn's format
    y_prob = np.column_stack([1-y_pred_proba, y_pred_proba])

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
        auc_roc = roc_auc_score(y_encoded, y_pred_proba)
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
        brier_score = brier_score_loss(y_encoded, y_pred_proba)
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

# LightGBM-specific metrics
lgb_metrics = {}

# Get model parameters
lgb_metrics["num_trees"] = model.num_trees()
lgb_metrics["num_features"] = model.num_feature()

# Check if model uses neural network components
model_str = model.model_to_string()
lgb_metrics["is_neural_network_like"] = "linear_tree" in model_str or "gbdt_pred_contrib" in model_str

# Get feature importance
feature_importance = {}

# Get feature importance by split
split_importance = model.feature_importance(importance_type='split')
for i, importance in enumerate(split_importance):
    if i < len(X.columns):
        feature_importance[f"split_{X.columns[i]}"] = float(importance)

# Get feature importance by gain
gain_importance = model.feature_importance(importance_type='gain')
for i, importance in enumerate(gain_importance):
    if i < len(X.columns):
        feature_importance[f"gain_{X.columns[i]}"] = float(importance)

# Normalize feature importance
total_gain = sum(gain_importance)
if total_gain > 0:
    for i, importance in enumerate(gain_importance):
        if i < len(X.columns):
            feature_importance[f"norm_gain_{X.columns[i]}"] = float(importance) / total_gain

# Get neural network specific parameters
try:
    # Parse model string to extract parameters
    params_dict = {}
    for line in model_str.split('\n'):
        if '=' in line:
            key, value = line.split('=', 1)
            params_dict[key.strip()] = value.strip()
    
    # Check for neural network related parameters
    if 'boosting_type' in params_dict:
        lgb_metrics["boosting_type"] = params_dict['boosting_type']
    
    if 'objective' in params_dict:
        lgb_metrics["objective"] = params_dict['objective']
    
    # Check for linear tree parameters
    if 'linear_tree' in params_dict:
        lgb_metrics["linear_tree"] = params_dict['linear_tree'] == 'true'
    
    # Check for dropout parameters
    if 'drop_rate' in params_dict:
        lgb_metrics["drop_rate"] = float(params_dict['drop_rate'])
    
    # Check for other neural network related parameters
    nn_params = ['max_drop', 'drop_seed', 'xgboost_dart_mode', 'uniform_drop']
    for param in nn_params:
        if param in params_dict:
            lgb_metrics[param] = params_dict[param]
    
except Exception as e:
    print(f"Warning: Could not extract all model parameters: {e}")

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

# Print LightGBM-specific metrics
print("\nLightGBM-specific metrics:")
print(f"Number of trees: {lgb_metrics['num_trees']}")
print(f"Number of features: {lgb_metrics['num_features']}")
if lgb_metrics.get("boosting_type"):
    print(f"Boosting type: {lgb_metrics['boosting_type']}")
if lgb_metrics.get("is_neural_network_like"):
    print("Model uses neural network-like architecture")
    if lgb_metrics.get("linear_tree"):
        print("Model uses linear trees")
    if lgb_metrics.get("drop_rate"):
        print(f"Dropout rate: {lgb_metrics['drop_rate']}")

# Print top feature importances
print("\nTop feature importances (gain):")
gain_features = {col: importance for col, importance in feature_importance.items() if col.startswith("gain_")}
sorted_importance = sorted(gain_features.items(), key=lambda x: x[1], reverse=True)
for feature, importance in sorted_importance[:10]:  # Show top 10
    print(f"{feature[5:]}: {importance:.4f}")  # Remove "gain_" prefix

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
    
    # Log LightGBM-specific metrics
    for metric_name, value in lgb_metrics.items():
        if isinstance(value, (int, float, bool)):
            mlflow.log_metric(metric_name, value)
        else:
            mlflow.log_param(metric_name, value)
    
    # Log model parameters
    try:
        # Extract parameters from model string
        params_dict = {}
        for line in model_str.split('\n'):
            if '=' in line:
                key, value = line.split('=', 1)
                params_dict[key.strip()] = value.strip()
        
        # Log parameters
        for param_name, param_value in params_dict.items():
            # Try to convert to appropriate type
            try:
                if param_value.lower() == 'true':
                    param_value = True
                elif param_value.lower() == 'false':
                    param_value = False
                elif '.' in param_value and param_value.replace('.', '', 1).isdigit():
                    param_value = float(param_value)
                elif param_value.isdigit():
                    param_value = int(param_value)
            except:
                pass
            
            mlflow.log_param(f"lgb_{param_name}", param_value)
    except Exception as e:
        print(f"Warning: Could not log all model parameters: {e}")
    
    # You can log additional information if needed
    mlflow.log_param("model_path", weight_path)
    mlflow.log_param("dataset_path", dataset_path)
    mlflow.log_param("num_classes", num_classes)
    
    # Log the model
    try:
        # Save model as a text file for visualization
        with open("model_text.txt", "w") as f:
            f.write(model_str)
        mlflow.log_artifact("model_text.txt")
        
        # Log the LightGBM model
        mlflow.lightgbm.log_model(model, "model")
    except Exception as e:
        print(f"Warning: Could not log model to MLflow: {e}")
    
    print(f"Metrics successfully logged to MLflow experiment: {experiment_name}")