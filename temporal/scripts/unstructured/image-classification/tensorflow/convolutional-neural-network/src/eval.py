import os
import zipfile
import shutil
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import (
    classification_report,
    log_loss,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from dotenv import load_dotenv
import mlflow
import traceback

# Load environment variables
load_dotenv()

model_path = os.getenv("MODEL_WIGHTS_PATH")
dataset_zip = os.getenv("DATASET_PATH")
experiment_name = os.getenv("EXPERIMENT_NAME")
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")

print(f"Using experiment: {experiment_name}")
print("MLflow URI:", mlflow_uri)
print("Experiment Name:", experiment_name)

# Step 1: Validate and Extract Dataset
extract_dir = "./temp_images"
if os.path.exists(extract_dir):
    shutil.rmtree(extract_dir)
os.makedirs(extract_dir, exist_ok=True)

try:
    if not dataset_zip.lower().endswith(".zip"):
        raise ValueError(f"Provided dataset path is not a .zip file: {dataset_zip}")
    
    with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"✅ Extracted dataset from {dataset_zip}")

    # Handle extra root directory inside ZIP (e.g., "test/")
    extracted_items = os.listdir(extract_dir)
    if len(extracted_items) == 1:
        nested = os.path.join(extract_dir, extracted_items[0])
        if os.path.isdir(nested):
            extract_dir = nested
            print(f"📂 Adjusted extract directory to: {extract_dir}")
except Exception as e:
    print(f"❌ Failed to extract dataset: {e}")
    traceback.print_exc()
    raise

# Step 2: Load model
try:
    model = load_model(model_path)
    input_shape = model.input_shape[1:3]
    print(f"✅ Loaded model from {model_path}")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    traceback.print_exc()
    raise

# Step 3: Prepare class label mapping
class_names = sorted([
    d for d in os.listdir(extract_dir)
    if os.path.isdir(os.path.join(extract_dir, d))
])
if not class_names:
    raise ValueError("❌ No class folders found in dataset. Make sure the ZIP has the correct structure.")

label_to_index = {name: idx for idx, name in enumerate(class_names)}
index_to_label = {idx: name for name, idx in label_to_index.items()}

# Step 4: Predict
y_true = []
y_pred = []

print(f"🖼️ Found {len(class_names)} classes: {class_names}")

for class_name in class_names:
    class_dir = os.path.join(extract_dir, class_name)
    true_label = label_to_index[class_name]

    for img_name in os.listdir(class_dir):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        img_path = os.path.join(class_dir, img_name)
        try:
            img = Image.open(img_path).resize(input_shape).convert("RGB")
            img_array = np.array(img)
            img_array = preprocess_input(img_array)
            img_array = np.expand_dims(img_array, axis=0)

            predictions = model.predict(img_array, verbose=0)
            pred_label = int(np.argmax(predictions))

            y_true.append(true_label)
            y_pred.append(pred_label)
        except Exception as e:
            print(f"⚠️ Skipping image {img_name}: {e}")

if not y_true:
    raise ValueError("❌ No valid images were processed. Please check the dataset content and structure.")

# Step 5: Evaluation
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
logloss = log_loss(y_true, to_categorical(y_pred, num_classes=len(class_names)))

print("\n📊 Evaluation Metrics:")
print(f"Accuracy     : {acc:.4f}")
print(f"Precision    : {prec:.4f}")
print(f"Recall       : {rec:.4f}")
print(f"F1 Score     : {f1:.4f}")
print(f"Log Loss     : {logloss:.4f}")
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

# Step 6: Log metrics to MLflow
mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_experiment(experiment_name)

try:
    with mlflow.start_run():
        mlflow.log_param("model_path", model_path)
        mlflow.log_param("dataset_zip", dataset_zip)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("log_loss", logloss)
        print("\n✅ Metrics logged to MLflow.")
except Exception as e:
    print(f"❌ MLflow logging failed: {e}")

print("🎉 Script completed successfully.")

# Cleanup
shutil.rmtree(extract_dir)
print("🧹 Cleaned up temporary extracted files.")
