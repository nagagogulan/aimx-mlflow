# src/eval.py

import os
import zipfile
import traceback
import shutil
import numpy as np
import pandas as pd
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, log_loss, accuracy_score, precision_score, recall_score, f1_score
from dotenv import load_dotenv
import mlflow
import logging

# Load environment variables
load_dotenv()

model_path = os.getenv("MODEL_WIGHTS_PATH")
dataset_zip = os.getenv("DATASET_PATH")
experiment_name = os.getenv("EXPERIMENT_NAME")
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")

# Step 1: Extract images
extract_dir = "./temp_images"
if os.path.exists(extract_dir):
    shutil.rmtree(extract_dir)
with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

# Step 2: Load model
model = load_model(model_path)
input_shape = model.input_shape[1:3]

# Step 3: Load labels.csv (should match images in zip)
labels_df = pd.read_csv(os.path.join("datasets", "labels.csv"))  # e.g., image_name,label
labels_dict = dict(zip(labels_df.image_name, labels_df.label))

# Step 4: Predict
y_true = []
y_pred = []

for img_name in os.listdir(extract_dir):
    if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
        continue
    img_path = os.path.join(extract_dir, img_name)
    img = Image.open(img_path).resize(input_shape).convert("RGB")
    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    pred_label = int(np.argmax(predictions))

    true_label = labels_dict.get(img_name, None)
    if true_label is not None:
        y_true.append(int(true_label))
        y_pred.append(pred_label)

# Step 5: Evaluate
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average="macro")
rec = recall_score(y_true, y_pred, average="macro")
f1 = f1_score(y_true, y_pred, average="macro")
logloss = log_loss(y_true, to_categorical(y_pred))

# Step 6: Print results
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Log Loss: {logloss:.4f}")
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred))

# Step 7: Log to MLflow
mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_experiment(experiment_name)

with mlflow.start_run():
    mlflow.log_param("model_path", model_path)
    mlflow.log_param("dataset_path", dataset_zip)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("log_loss", logloss)

    print("\n✅ Metrics logged to MLflow")

# Cleanup
shutil.rmtree(extract_dir)
