import os
import pandas as pd
import numpy as np
import joblib
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

# Step 4: Load your trained model
model = joblib.load(weight_path)

# Step 5: Predict
predictions = model.predict(X)

# Get prediction probabilities for ROC-AUC calculation
# Check if the model has predict_proba method
try:
    # For multi-class classification
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X)
    # For models that only have decision_function (like SVM)
    elif hasattr(model, 'decision_function'):
        y_prob = model.decision_function(X)
        # Reshape for binary classification
        if len(y_prob.shape) == 1:
            y_prob = np.column_stack([1-y_prob, y_prob])
    else:
        y_prob = None
except:
    y_prob = None

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
    if y_prob is not None:
        if num_classes == 2:
            # Binary classification
            auc_roc = roc_auc_score(y_encoded, y_prob[:, 1])
        else:
            # Multi-class: use one-vs-rest approach (ovr)
            y_one_hot = np.zeros((len(y_encoded), num_classes))
            for i in range(len(y_encoded)):
                y_one_hot[i, y_encoded[i]] = 1
            auc_roc = roc_auc_score(y_one_hot, y_prob, multi_class='ovr')
except Exception as e:
    print(f"Warning: Could not calculate AUC-ROC: {e}")

# Calculate Logarithmic Loss (LogLoss)
logloss = 0  # Default value
try:
    if y_prob is not None:
        if num_classes == 2:
            # Binary classification
            logloss = log_loss(y_encoded, y_prob[:, 1])
        else:
            # Multi-class
            y_one_hot = np.zeros((len(y_encoded), num_classes))
            for i in range(len(y_encoded)):
                y_one_hot[i, y_encoded[i]] = 1
            logloss = log_loss(y_one_hot, y_prob)
except Exception as e:
    print(f"Warning: Could not calculate LogLoss: {e}")

# Calculate Brier Score: (1/N) Σ (y_i - ŷ_i)^2
brier_score = 0  # Default value
try:
    if y_prob is not None and num_classes == 2:
        # Binary classification
        brier_score = brier_score_loss(y_encoded, y_prob[:, 1])
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

# Scikit-learn neural network specific metrics
sklearn_metrics = {}

# Check if model is a neural network (MLPClassifier)
is_neural_network = False
try:
    from sklearn.neural_network import MLPClassifier
    is_neural_network = isinstance(model, MLPClassifier)
except:
    pass

# Get neural network specific parameters if available
if is_neural_network:
    sklearn_metrics["is_neural_network"] = True
    
    # Get number of layers and neurons per layer
    if hasattr(model, 'n_layers_'):
        sklearn_metrics["n_layers"] = model.n_layers_
    
    if hasattr(model, 'hidden_layer_sizes'):
        sklearn_metrics["hidden_layer_sizes"] = str(model.hidden_layer_sizes)
    
    # Get activation function
    if hasattr(model, 'activation'):
        sklearn_metrics["activation_function"] = model.activation
    
    # Get solver (optimization algorithm)
    if hasattr(model, 'solver'):
        sklearn_metrics["solver"] = model.solver
    
    # Get learning rate and schedule
    if hasattr(model, 'learning_rate'):
        sklearn_metrics["learning_rate_type"] = model.learning_rate
    
    if hasattr(model, 'learning_rate_init'):
        sklearn_metrics["initial_learning_rate"] = float(model.learning_rate_init)
    
    # Get regularization parameters
    if hasattr(model, 'alpha'):
        sklearn_metrics["alpha_regularization"] = float(model.alpha)
    
    # Get batch size
    if hasattr(model, 'batch_size'):
        sklearn_metrics["batch_size"] = model.batch_size
    
    # Get number of iterations
    if hasattr(model, 'n_iter_'):
        sklearn_metrics["n_iterations"] = int(model.n_iter_)
    
    # Get convergence info
    if hasattr(model, 'loss_'):
        sklearn_metrics["final_loss"] = float(model.loss_)
    
    # Get number of outputs
    if hasattr(model, 'n_outputs_'):
        sklearn_metrics["n_outputs"] = int(model.n_outputs_)
    
    # Get weights and biases sizes
    if hasattr(model, 'coefs_'):
        sklearn_metrics["n_weights"] = sum(w.size for w in model.coefs_)
    
    if hasattr(model, 'intercepts_'):
        sklearn_metrics["n_biases"] = sum(b.size for b in model.intercepts_)
else:
    sklearn_metrics["is_neural_network"] = False
    
    # For other models, get available metrics
    # For ensemble models, get number of estimators
    if hasattr(model, 'n_estimators'):
        sklearn_metrics["n_estimators"] = model.n_estimators
    
    # For tree-based models, get tree info
    if hasattr(model, 'tree_'):
        sklearn_metrics["max_depth"] = model.tree_.max_depth
        sklearn_metrics["node_count"] = model.tree_.node_count
    
    # For linear models, get coefficients
    if hasattr(model, 'coef_'):
        sklearn_metrics["n_coefficients"] = model.coef_.size

# Get feature importance if available
feature_importance = {}
if hasattr(model, 'feature_importances_'):
    for i, importance in enumerate(model.feature_importances_):
        feature_importance[X.columns[i]] = float(importance)
elif hasattr(model, 'coef_') and len(model.coef_.shape) == 2:
    # For linear models with multiple classes
    for class_idx in range(model.coef_.shape[0]):
        for i, coef in enumerate(model.coef_[class_idx]):
            feature_importance[f"class{class_idx}_{X.columns[i]}"] = float(coef)
elif hasattr(model, 'coef_') and len(model.coef_.shape) == 1:
    # For linear models with binary classification
    for i, coef in enumerate(model.coef_):
        feature_importance[X.columns[i]] = float(coef)

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

# Print scikit-learn specific metrics
print("\nScikit-learn specific metrics:")
if sklearn_metrics.get("is_neural_network", False):
    print("Model is a neural network (MLPClassifier)")
    if "n_layers" in sklearn_metrics:
        print(f"Number of layers: {sklearn_metrics['n_layers']}")
    if "hidden_layer_sizes" in sklearn_metrics:
        print(f"Hidden layer sizes: {sklearn_metrics['hidden_layer_sizes']}")
    if "activation_function" in sklearn_metrics:
        print(f"Activation function: {sklearn_metrics['activation_function']}")
    if "solver" in sklearn_metrics:
        print(f"Solver: {sklearn_metrics['solver']}")
    if "n_iterations" in sklearn_metrics:
        print(f"Number of iterations: {sklearn_metrics['n_iterations']}")
    if "final_loss" in sklearn_metrics:
        print(f"Final loss: {sklearn_metrics['final_loss']}")
    if "n_weights" in sklearn_metrics and "n_biases" in sklearn_metrics:
        print(f"Network size: {sklearn_metrics['n_weights']} weights, {sklearn_metrics['n_biases']} biases")
else:
    print("Model is not a neural network")
    for metric, value in sklearn_metrics.items():
        if metric != "is_neural_network":
            print(f"{metric}: {value}")

# Print top feature importances if available
if feature_importance:
    print("\nTop feature importances:")
    sorted_importance = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
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
    
    # Log scikit-learn specific metrics
    for metric_name, value in sklearn_metrics.items():
        if isinstance(value, (int, float, bool)):
            mlflow.log_metric(metric_name, value)
        else:
            mlflow.log_param(metric_name, value)
    
    # Log model parameters
    if hasattr(model, 'get_params'):
        params = model.get_params()
        for param_name, param_value in params.items():
            if isinstance(param_value, (int, float, str, bool)):
                mlflow.log_param(param_name, param_value)
    
    # You can log additional information if needed
    mlflow.log_param("model_path", weight_path)
    mlflow.log_param("dataset_path", dataset_path)
    mlflow.log_param("num_classes", num_classes)
    
    # Log the model
    try:
        # Log the scikit-learn model
        mlflow.sklearn.log_model(model, "model")
    except Exception as e:
        print(f"Warning: Could not log model to MLflow: {e}")
    
    print(f"Metrics successfully logged to MLflow experiment: {experiment_name}")