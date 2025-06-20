import os
import pandas as pd
import numpy as np
import xgboost as xgb
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

# Step 4: Load your trained XGBoost model
model = xgb.Booster()
model.load_model(weight_path)

# Convert data to DMatrix format for XGBoost
dtest = xgb.DMatrix(X)

# Step 5: Predict
# Get raw predictions (margin values)
margin_pred = model.predict(dtest, output_margin=True)

# For multi-class classification
if num_classes > 2:
    # XGBoost outputs a flattened array for multi-class, reshape it
    margin_pred = margin_pred.reshape(X.shape[0], num_classes)
    # Convert margins to probabilities using softmax
    y_pred_proba = np.exp(margin_pred) / np.sum(np.exp(margin_pred), axis=1, keepdims=True)
    # Get class with highest probability
    y_pred = np.argmax(y_pred_proba, axis=1)
else:
    # For binary classification
    # Convert margin to probability using sigmoid
    y_pred_proba = 1.0 / (1.0 + np.exp(-margin_pred))
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

# XGBoost-specific metrics
xgb_metrics = {}

# Get model parameters
xgb_metrics["model_params"] = model.get_params()

# Get number of trees
xgb_metrics["num_trees"] = model.num_boosted_rounds()

# Get feature importance
feature_importance = {}
importance_scores = model.get_score(importance_type='gain')
for feature, score in importance_scores.items():
    # Map feature index to feature name if needed
    if feature.startswith('f'):
        try:
            idx = int(feature[1:])
            if idx < len(X.columns):
                feature = X.columns[idx]
        except:
            pass
    feature_importance[feature] = score

# Normalize feature importance
if importance_scores:
    total_importance = sum(importance_scores.values())
    for feature, score in feature_importance.items():
        feature_importance[feature] = score / total_importance

# Get SHAP values for explainability (if available)
shap_values = None
try:
    # Predict SHAP values
    shap_values = model.predict(dtest, pred_contribs=True)
    # For binary classification, we only need the last column (excluding bias)
    if num_classes == 2 and shap_values.shape[1] > 1:
        shap_values = shap_values[:, :-1]  # Remove bias term
    
    # Calculate mean absolute SHAP value for each feature
    mean_shap_values = np.abs(shap_values).mean(axis=0)
    
    # Create a dictionary of feature names to mean SHAP values
    shap_importance = {}
    for i, col in enumerate(X.columns):
        if i < len(mean_shap_values):
            shap_importance[col] = float(mean_shap_values[i])
    
    # Add to metrics
    xgb_metrics["shap_available"] = True
except Exception as e:
    print(f"Warning: Could not calculate SHAP values: {e}")
    xgb_metrics["shap_available"] = False

# Get neural network specific parameters (if using gblinear or dart booster)
booster_type = model.get_params().get('booster', '')
if booster_type in ['gblinear', 'dart']:
    xgb_metrics["is_neural_network_like"] = True
    
    # For gblinear, get the linear weights
    if booster_type == 'gblinear':
        try:
            # Get the linear weights
            weights = model.get_dump()
            xgb_metrics["linear_weights_available"] = True
        except:
            xgb_metrics["linear_weights_available"] = False
    
    # For dart, get dropout information
    if booster_type == 'dart':
        try:
            rate_drop = model.get_params().get('rate_drop', 0)
            skip_drop = model.get_params().get('skip_drop', 0)
            xgb_metrics["dropout_rate"] = rate_drop
            xgb_metrics["skip_dropout_probability"] = skip_drop
        except:
            pass
else:
    xgb_metrics["is_neural_network_like"] = False

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

# Print XGBoost-specific metrics
print("\nXGBoost-specific metrics:")
print(f"Number of trees: {xgb_metrics['num_trees']}")
print(f"Booster type: {booster_type}")
if xgb_metrics.get("is_neural_network_like", False):
    print("Model uses neural network-like architecture")
    if booster_type == 'dart':
        print(f"Dropout rate: {xgb_metrics.get('dropout_rate', 'N/A')}")
        print(f"Skip dropout probability: {xgb_metrics.get('skip_dropout_probability', 'N/A')}")

# Print top feature importances
print("\nTop feature importances (gain):")
sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
for feature, importance in sorted_importance[:10]:  # Show top 10
    print(f"{feature}: {importance:.4f}")

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
    
    # Log XGBoost-specific metrics
    for metric_name, value in xgb_metrics.items():
        if isinstance(value, (int, float, bool)):
            mlflow.log_metric(metric_name, value)
        elif metric_name != "model_params":  # Don't log the full model params
            mlflow.log_param(metric_name, value)
    
    # Log model parameters
    if "model_params" in xgb_metrics:
        for param_name, param_value in xgb_metrics["model_params"].items():
            if isinstance(param_value, (int, float, str, bool)):
                mlflow.log_param(f"xgb_{param_name}", param_value)
    
    # Log SHAP values if available
    if shap_values is not None:
        # Log mean absolute SHAP values for each feature
        for feature, importance in shap_importance.items():
            mlflow.log_metric(f"shap_{feature}", importance)
    
    # You can log additional information if needed
    mlflow.log_param("model_path", weight_path)
    mlflow.log_param("dataset_path", dataset_path)
    mlflow.log_param("num_classes", num_classes)
    mlflow.log_param("booster_type", booster_type)
    
    # Log the model
    try:
        # Save model as a JSON file for visualization
        model_json = model.save_config()
        with open("model_config.json", "w") as f:
            f.write(model_json)
        mlflow.log_artifact("model_config.json")
        
        # Log the XGBoost model
        mlflow.xgboost.log_model(model, "model")
    except Exception as e:
        print(f"Warning: Could not log model to MLflow: {e}")
    
    print(f"Metrics successfully logged to MLflow experiment: {experiment_name}")