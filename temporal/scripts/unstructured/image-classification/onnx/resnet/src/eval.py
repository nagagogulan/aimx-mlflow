import os
import csv
from dotenv import load_dotenv
import onnxruntime as ort
import numpy as np
from PIL import Image
import shutil
import zipfile
import mlflow

# Load variables from .env file
load_dotenv()

weight_path = os.getenv("MODEL_WIGHTS_PATH")
images_zip_path = os.getenv("IMAGES_ZIP_PATH")
labels_path = os.getenv("DATA_LABELS_PATH")
dataset_path = os.getenv("DATASET_PATH")
mlflowURI = os.getenv("MLFLOW_TRACKING_URI")
experiment_name = os.getenv("EXPERIMENT_NAME")

# ----------------------------
# Helper Functions
# ----------------------------
def preprocess_image(image_path, input_size=(224, 224)):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(input_size)
    image_data = np.array(image).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_data = (image_data - mean) / std
    image_data = np.transpose(image_data, (2, 0, 1))  # Channels first
    image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
    return image_data

def load_labels_csv(csv_path):
    label_map = {}
    with open(csv_path, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = os.path.basename(row['image_path'])  # Just the filename
            label_map[path] = int(row['label'])
    return label_map

# ----------------------------
# Load ONNX Model
# ----------------------------
session = ort.InferenceSession(weight_path)

# ----------------------------
# Extract Images
# ----------------------------
with zipfile.ZipFile(images_zip_path, 'r') as zip_ref:
    zip_ref.extractall("./temp_images")

image_dir = "./temp_images"
image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# ----------------------------
# Load Ground Truth Labels
# ----------------------------
labels = load_labels_csv(dataset_path)

# ----------------------------
# Run Inference & Evaluate
# ----------------------------
correct = 0
total = 0
class_tp = {}  # True positives per class
class_fp = {}  # False positives per class
class_fn = {}  # False negatives per class
class_count = {}  # Total instances per class

for image_file in image_files:
    input_data = preprocess_image(image_file)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_data})
    predictions = outputs[0]
    top_class = int(np.argmax(predictions))

    image_name = os.path.basename(image_file)
    true_label = labels.get(image_name)

    if true_label is not None:
        total += 1
        # Count instances per class
        class_count[true_label] = class_count.get(true_label, 0) + 1
        
        if top_class == true_label:
            correct += 1
            # True positive for the predicted class
            class_tp[top_class] = class_tp.get(top_class, 0) + 1
        else:
            # False positive for the predicted class
            class_fp[top_class] = class_fp.get(top_class, 0) + 1
            # False negative for the true class
            class_fn[true_label] = class_fn.get(true_label, 0) + 1
        
        print(f"Image: {image_name} -> Predicted: {top_class}, Actual: {true_label}")
    else:
        print(f"[⚠️ Warning] No label found for image: {image_name}")

# ----------------------------
# Print Accuracy
# ----------------------------
accuracy = correct / total if total > 0 else 0
print(f"\n✅ Accuracy: {accuracy * 100:.2f}% ({correct}/{total})")

# Calculate recall for each class
class_recall = {}
for cls in set(class_tp.keys()) | set(class_fn.keys()):
    tp = class_tp.get(cls, 0)
    fn = class_fn.get(cls, 0)
    # Recall = TP / (TP + FN)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    class_recall[cls] = recall
    print(f"Class {cls} Recall: {recall * 100:.2f}% (TP: {tp}, FN: {fn})")

# Calculate macro-average recall
if class_recall:
    macro_recall = sum(class_recall.values()) / len(class_recall)
    print(f"\n📊 Macro-Average Recall: {macro_recall * 100:.2f}%")

# ----------------------------
# Calculate and Print Precision
# ----------------------------
# Calculate precision for each class
class_precision = {}
for cls in set(class_tp.keys()) | set(class_fp.keys()):
    tp = class_tp.get(cls, 0)
    fp = class_fp.get(cls, 0)
    # Precision = TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    class_precision[cls] = precision
    print(f"Class {cls} Precision: {precision * 100:.2f}% (TP: {tp}, FP: {fp})")

# Calculate macro-average precision (average of precision for all classes)
if class_precision:
    macro_precision = sum(class_precision.values()) / len(class_precision)
    print(f"\n📊 Macro-Average Precision: {macro_precision * 100:.2f}%")

# ----------------------------
# Calculate and Print F1-Score
# ----------------------------
# Calculate F1-score for each class
class_f1 = {}
all_classes = set(class_precision.keys()) | set(class_recall.keys())
for cls in all_classes:
    precision = class_precision.get(cls, 0)
    recall = class_recall.get(cls, 0)
    # F1-score = 2 * (Precision * Recall) / (Precision + Recall)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    class_f1[cls] = f1
    print(f"Class {cls} F1-score: {f1 * 100:.2f}%")

# Calculate macro-average F1-score
if class_f1:
    macro_f1 = sum(class_f1.values()) / len(class_f1)
    print(f"\n📊 Macro-Average F1-score: {macro_f1 * 100:.2f}%")

# ----------------------------
# Calculate and Print Balanced Accuracy
# ----------------------------
# Calculate specificity for each class (TN / (TN + FP))
class_specificity = {}
all_classes = set(class_tp.keys()) | set(class_fp.keys()) | set(class_fn.keys())
total_samples = total

for cls in all_classes:
    # For each class, all other class instances are negatives
    # True Negatives: samples of other classes correctly not predicted as this class
    # Sum of all classes' samples minus (TP + FN) for this class
    tp = class_tp.get(cls, 0)
    fp = class_fp.get(cls, 0)
    fn = class_fn.get(cls, 0)
    
    # TN = total - (TP + FP + FN)
    tn = total_samples - (tp + fp + fn)
    
    # Specificity = TN / (TN + FP)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    class_specificity[cls] = specificity
    
    # Get recall (sensitivity) for this class
    sensitivity = class_recall.get(cls, 0)
    
    # Balanced accuracy = (Sensitivity + Specificity) / 2
    balanced_acc = (sensitivity + specificity) / 2
    print(f"Class {cls} Balanced Accuracy: {balanced_acc * 100:.2f}% (Sensitivity: {sensitivity * 100:.2f}%, Specificity: {specificity * 100:.2f}%)")

# Calculate macro-average balanced accuracy
if class_specificity:
    # Combine sensitivity (recall) and specificity for each class
    class_balanced_acc = {}
    for cls in all_classes:
        sensitivity = class_recall.get(cls, 0)
        specificity = class_specificity.get(cls, 0)
        class_balanced_acc[cls] = (sensitivity + specificity) / 2
    
    macro_balanced_acc = sum(class_balanced_acc.values()) / len(class_balanced_acc)
    print(f"\n📊 Macro-Average Balanced Accuracy: {macro_balanced_acc * 100:.2f}%")

# ----------------------------
# Setup MLflow Tracking
# ----------------------------
mlflow.set_tracking_uri(mlflowURI)
mlflow.set_experiment(experiment_name)

# Start an MLflow run
with mlflow.start_run():
    # Log model file as an artifact
    mlflow.log_artifact(weight_path, "model")
    
    # Log overall metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("macro_precision", macro_precision if 'macro_precision' in locals() else 0)
    mlflow.log_metric("macro_recall", macro_recall if 'macro_recall' in locals() else 0)
    mlflow.log_metric("macro_f1", macro_f1 if 'macro_f1' in locals() else 0)
    mlflow.log_metric("macro_balanced_accuracy", macro_balanced_acc if 'macro_balanced_acc' in locals() else 0)
    
    # Log per-class metrics
    for cls in all_classes:
        prefix = f"class_{cls}_"
        if cls in class_precision:
            mlflow.log_metric(f"{prefix}precision", class_precision[cls])
        if cls in class_recall:
            mlflow.log_metric(f"{prefix}recall", class_recall[cls])
        if cls in class_f1:
            mlflow.log_metric(f"{prefix}f1", class_f1[cls])
        if cls in class_balanced_acc:
            mlflow.log_metric(f"{prefix}balanced_acc", class_balanced_acc[cls])
        if cls in class_specificity:
            mlflow.log_metric(f"{prefix}specificity", class_specificity[cls])
        
        # Log raw counts
        mlflow.log_metric(f"{prefix}tp", class_tp.get(cls, 0))
        mlflow.log_metric(f"{prefix}fp", class_fp.get(cls, 0))
        mlflow.log_metric(f"{prefix}fn", class_fn.get(cls, 0))
        mlflow.log_metric(f"{prefix}count", class_count.get(cls, 0))
    
    # Log log loss if calculated
    if 'avg_log_loss' in locals():
        mlflow.log_metric("log_loss", avg_log_loss)
    
    # Log parameters
    mlflow.log_param("model_path", weight_path)
    mlflow.log_param("total_samples", total)
    mlflow.log_param("num_classes", len(all_classes))
    
    print("\n✅ All metrics have been logged to MLflow")

# ----------------------------
# Calculate and Print Log Loss
# ----------------------------
# We need to track all predictions and true labels for log loss
log_loss_sum = 0
valid_samples = 0

# Re-run predictions to get probability scores for log loss calculation
for image_file in image_files:
    image_name = os.path.basename(image_file)
    true_label = labels.get(image_name)
    
    if true_label is not None:
        input_data = preprocess_image(image_file)
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: input_data})
        
        # Get probabilities using softmax
        probabilities = np.exp(outputs[0][0]) / np.sum(np.exp(outputs[0][0]))
        
        # Extract the predicted probability for the true class
        true_class_prob = probabilities[true_label]
        
        # Clip the probability to avoid log(0)
        epsilon = 1e-15
        true_class_prob = max(epsilon, min(1 - epsilon, true_class_prob))
        
        # Calculate log loss for this sample: -log(true_class_prob)
        sample_loss = -np.log(true_class_prob)
        log_loss_sum += sample_loss
        valid_samples += 1

# Calculate average log loss
if valid_samples > 0:
    avg_log_loss = log_loss_sum / valid_samples
    print(f"\n📉 Log Loss: {avg_log_loss:.4f}")

# ----------------------------
# Cleanup Temporary Directory
# ----------------------------
shutil.rmtree(image_dir)
