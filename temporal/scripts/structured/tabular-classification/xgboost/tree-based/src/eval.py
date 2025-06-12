import os
import traceback
import pandas as pd
import joblib
import mlflow
import subprocess
import docker
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score, roc_curve, log_loss, cohen_kappa_score
from sklearn.metrics import brier_score_loss

# Load variables from .env file
load_dotenv()


# Step 0: Connect to Minikube Docker daemon
try:
    print("🔌 Connecting to Minikube Docker daemon...", flush=True)
    docker_env = subprocess.check_output("minikube docker-env --shell bash", shell=True).decode()
    for line in docker_env.splitlines():
        if line.startswith("export "):
            key, val = line.replace("export ", "").split("=", 1)
            os.environ[key] = val.strip('"')

    client = docker.from_env()
    print("✅ Connected to Minikube Docker daemon", flush=True)

    # Step 1: Check for container
    containers = client.containers.list(all=True, filters={"ancestor": "aimx-evaluation:latest"})
    for container in containers:
        print(f"🛑 Stopping and removing container: {container.name}", flush=True)
        container.stop()
        container.remove(force=True)

    # Step 2: Check and remove image
    images = client.images.list(name="aimx-evaluation:latest")
    for image in images:
        print(f"🧼 Removing image: {image.short_id}", flush=True)
        client.images.remove(image.id, force=True)

    print("🚮 Docker cleanup complete (container + image)", flush=True)

except Exception as e:
    print(f"⚠️ Failed to connect or clean Docker resources: {e}", flush=True)
    traceback.print_exc()

mlflowURI = os.getenv("MLFLOW_TRACKING_URI")
weight_path = os.path.abspath(os.getenv("MODEL_WIGHTS_PATH"))
dataset_path = os.path.abspath(os.getenv("DATASET_PATH"))
target_column = os.getenv("TARGET_COLUMN")
experiment_name = os.getenv("EXPERIMENT_NAME")

# Now proceed to load CSV inside try-except
try:
    df = pd.read_csv(dataset_path)
    print(f"✅ Loaded dataset with shape: {df.shape}", flush=True)
except Exception as e:
    print(f"❌ Failed to load dataset from: {dataset_path}", flush=True)
    traceback.print_exc()
    raise

# Step 2: Separate features and target
X = df.drop(columns=[target_column])
y = df[target_column]

# Step 3: Preprocess if needed
# Convert object-type features to category
for col in X.select_dtypes(include='object').columns:
    X[col] = X[col].astype('category')

# Step 4: Load your trained model
model = joblib.load(weight_path)  # Update if filename differs

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
    # For binary classification or if classes are encoded as 0 and 1
    if y_prob is not None:
        if len(model.classes_) == 2:
            # Binary classification: use probabilities of the positive class
            auc_roc = roc_auc_score(y, y_prob[:, 1])
        else:
            # Multi-class: use one-vs-rest approach (ovr)
            auc_roc = roc_auc_score(pd.get_dummies(y), y_prob, multi_class='ovr')
except Exception as e:
    print(f"Warning: Could not calculate AUC-ROC: {e}")

# Calculate Logarithmic Loss (LogLoss)
# Formula: -(1/N) Σ [y_i log(ŷ_i) + (1-y_i) log(1-ŷ_i)]
logloss = 0  # Default value
try:
    if y_prob is not None:
        # Calculate log loss using scikit-learn's log_loss function
        logloss = log_loss(y, y_prob)
except Exception as e:
    print(f"Warning: Could not calculate LogLoss: {e}")

# Calculate Brier Score: (1/N) Σ (y_i - ŷ_i)^2
brier_score = 0  # Default value
try:
    if y_prob is not None and len(model.classes_) == 2:
        # Convert target to binary format for binary classification
        y_binary = (y == model.classes_[1]).astype(int)
        # Calculate Brier score using scikit-learn's brier_score_loss function
        brier_score = brier_score_loss(y_binary, y_prob[:, 1])
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
    
    # You can log additional information if needed
    mlflow.log_param("model_path", weight_path)
    mlflow.log_param("dataset_path", dataset_path)
    
    print(f"Metrics successfully logged to MLflow experiment: {experiment_name}")
