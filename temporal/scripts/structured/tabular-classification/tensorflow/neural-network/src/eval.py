import os
import pandas as pd
import numpy as np
import tensorflow as tf
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

# Convert to numpy arrays
X_array = X.values.astype('float32')

# Step 4: Load your trained TensorFlow model
model = tf.keras.models.load_model(weight_path)

# Step 5: Predict
# Get raw predictions (logits)
y_pred_proba = model.predict(X_array)

# For binary classification, reshape if needed
if num_classes == 2 and len(y_pred_proba.shape) == 1:
    y_pred_proba = np.column_stack([1-y_pred_proba, y_pred_proba])

# Get class with highest probability
if num_classes > 2:
    y_pred = np.argmax(y_pred_proba, axis=1)
else:
    y_pred = (y_pred_proba[:, 1] > 0.5).astype(int) if y_pred_proba.shape[1] == 2 else (y_pred_proba > 0.5).astype(int)

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
        if y_pred_proba.shape[1] == 2:
            auc_roc = roc_auc_score(y_encoded, y_pred_proba[:, 1])
        else:
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
        if y_pred_proba.shape[1] == 2:
            logloss = log_loss(y_encoded, y_pred_proba[:, 1])
        else:
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
        if y_pred_proba.shape[1] == 2:
            brier_score = brier_score_loss(y_encoded, y_pred_proba[:, 1])
        else:
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

# TensorFlow-specific metrics
tf_metrics = {}

# Get model architecture summary
model_summary = []
model.summary(print_fn=lambda x: model_summary.append(x))
tf_metrics["model_summary"] = "\n".join(model_summary)

# Get number of layers
tf_metrics["num_layers"] = len(model.layers)

# Get number of trainable parameters
tf_metrics["trainable_params"] = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
tf_metrics["non_trainable_params"] = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])

# Get layer types
layer_types = {}
for layer in model.layers:
    layer_type = layer.__class__.__name__
    if layer_type in layer_types:
        layer_types[layer_type] += 1
    else:
        layer_types[layer_type] = 1
tf_metrics["layer_types"] = layer_types

# Get activation functions
activation_functions = set()
for layer in model.layers:
    if hasattr(layer, 'activation'):
        if hasattr(layer.activation, '__name__'):
            activation_functions.add(layer.activation.__name__)
        elif hasattr(layer.activation, 'name'):
            activation_functions.add(layer.activation.name)
tf_metrics["activation_functions"] = list(activation_functions)

# Get optimizer info if available
if hasattr(model, 'optimizer') and model.optimizer is not None:
    tf_metrics["optimizer"] = model.optimizer.__class__.__name__
    if hasattr(model.optimizer, 'get_config'):
        try:
            optimizer_config = model.optimizer.get_config()
            for key, value in optimizer_config.items():
                if isinstance(value, (int, float, str, bool)):
                    tf_metrics[f"optimizer_{key}"] = value
        except:
            pass

# Get loss function if available
if hasattr(model, 'loss'):
    if isinstance(model.loss, dict):
        tf_metrics["loss_function"] = str(model.loss)
    elif hasattr(model.loss, '__name__'):
        tf_metrics["loss_function"] = model.loss.__name__
    elif isinstance(model.loss, str):
        tf_metrics["loss_function"] = model.loss

# Get feature importance using permutation importance
# This is computationally expensive, so we'll limit it to a small number of features
feature_importance = {}
try:
    # Use a simple approach: measure change in prediction when each feature is zeroed out
    baseline_pred = model.predict(X_array)
    for i, feature_name in enumerate(X.columns):
        # Create a copy of the data
        X_modified = X_array.copy()
        # Zero out the feature
        X_modified[:, i] = 0
        # Get new predictions
        new_pred = model.predict(X_modified)
        # Calculate change in prediction (mean absolute difference)
        importance = np.mean(np.abs(baseline_pred - new_pred))
        feature_importance[feature_name] = float(importance)
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

# Print TensorFlow-specific metrics
print("\nTensorFlow-specific metrics:")
print(f"Number of layers: {tf_metrics['num_layers']}")
print(f"Trainable parameters: {tf_metrics['trainable_params']}")
print(f"Non-trainable parameters: {tf_metrics['non_trainable_params']}")
print("Layer types:")
for layer_type, count in layer_types.items():
    print(f"  {layer_type}: {count}")
print(f"Activation functions: {', '.join(tf_metrics['activation_functions'])}")
if "optimizer" in tf_metrics:
    print(f"Optimizer: {tf_metrics['optimizer']}")
if "loss_function" in tf_metrics:
    print(f"Loss function: {tf_metrics['loss_function']}")

# Print top feature importances
if feature_importance:
    print("\nTop feature importances:")
    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    for feature, importance in sorted_importance[:10]:  # Show top 10
        print(f"{feature}: {importance:.4f}")

# Print model summary
print("\nModel Summary:")
print(tf_metrics["model_summary"])

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
    
    # Log TensorFlow-specific metrics
    for metric_name, value in tf_metrics.items():
        if isinstance(value, (int, float)):
            mlflow.log_metric(metric_name, value)
        elif metric_name == "model_summary":
            # Log model summary as a text artifact
            with open("model_summary.txt", "w") as f:
                f.write(value)
            mlflow.log_artifact("model_summary.txt")
        elif isinstance(value, dict):
            # For dictionaries like layer_types
            for k, v in value.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"{metric_name}_{k}", v)
                else:
                    mlflow.log_param(f"{metric_name}_{k}", v)
        elif isinstance(value, list):
            # For lists like activation_functions
            mlflow.log_param(metric_name, ", ".join(map(str, value)))
        else:
            mlflow.log_param(metric_name, value)
    
    # You can log additional information if needed
    mlflow.log_param("model_path", weight_path)
    mlflow.log_param("dataset_path", dataset_path)
    mlflow.log_param("num_classes", num_classes)
    
    # Log the model architecture as a JSON file
    try:
        model_json = model.to_json()
        with open("model_architecture.json", "w") as f:
            f.write(model_json)
        mlflow.log_artifact("model_architecture.json")
    except:
        pass
    
    # Log the TensorFlow model
    try:
        mlflow.tensorflow.log_model(model, "model")
    except Exception as e:
        print(f"Warning: Could not log model to MLflow: {e}")
    
    print(f"Metrics successfully logged to MLflow experiment: {experiment_name}")