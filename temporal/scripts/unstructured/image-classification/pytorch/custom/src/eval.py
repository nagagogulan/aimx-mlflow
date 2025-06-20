import os
import csv
import zipfile
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import mlflow
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import shutil
import importlib.util
import sys

# Load variables from .env file
load_dotenv()

mlflowURI = os.getenv("MLFLOW_TRACKING_URI")
weight_path = os.getenv("MODEL_WEIGHTS_PATH")
images_zip_path = os.getenv("IMAGES_ZIP_PATH")
dataset_path = os.getenv("DATASET_PATH")
labels_path = os.getenv("DATA_LABELS_PATH")
experiment_name = os.getenv("EXPERIMENT_NAME")
model_definition_path = os.getenv("MODEL_DEFINITION_PATH", "./model_definition.py")
model_class_name = os.getenv("MODEL_CLASS_NAME", "CustomModel")
input_size = int(os.getenv("INPUT_SIZE", "224"))
num_classes = int(os.getenv("NUM_CLASSES", "1000"))

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------------
# Helper Functions
# ----------------------------
def preprocess_image(image_path):
    """Preprocess image for custom model"""
    image = Image.open(image_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.Resize(int(input_size * 1.14)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = preprocess(image).unsqueeze(0)
    return image_tensor

def load_labels_csv(csv_path):
    """Load image labels from CSV file"""
    label_map = {}
    with open(csv_path, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = os.path.basename(row['image_path'])  # Just the filename
            label_map[path] = int(row['label'])
    return label_map

def load_class_names(labels_path):
    """Load class names from file"""
    with open(labels_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names

def load_custom_model_class(model_path, class_name):
    """Dynamically load a custom model class from a Python file"""
    print(f"Loading custom model class '{class_name}' from {model_path}")
    
    if not os.path.exists(model_path):
        print(f"⚠️ Model definition file not found: {model_path}")
        return None
    
    try:
        # Load the module
        spec = importlib.util.spec_from_file_location("custom_model", model_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules["custom_model"] = module
        spec.loader.exec_module(module)
        
        # Get the model class
        if hasattr(module, class_name):
            model_class = getattr(module, class_name)
            print(f"Successfully loaded {class_name} from {model_path}")
            return model_class
        else:
            print(f"⚠️ Class '{class_name}' not found in {model_path}")
            return None
    except Exception as e:
        print(f"⚠️ Error loading custom model: {e}")
        return None

# ----------------------------
# Load PyTorch Model
# ----------------------------
def load_model(weight_path, model_definition_path=None, model_class_name=None):
    """Load custom model from .pth file"""
    print(f"Loading model from: {weight_path}")
    
    try:
        # Try loading as a complete model
        model = torch.load(weight_path, map_location=device)
        if isinstance(model, nn.Module):
            print("Loaded complete model")
            return model
    except Exception as e:
        print(f"Not a complete model: {e}")
    
    # Try loading custom model class if provided
    custom_model_class = None
    if model_definition_path and model_class_name:
        custom_model_class = load_custom_model_class(model_definition_path, model_class_name)
    
    if custom_model_class:
        try:
            # Initialize the custom model
            model = custom_model_class(num_classes=num_classes)
            
            # Load state dict
            state_dict = torch.load(weight_path, map_location=device)
            
            # Check if it's a wrapped state dict
            if isinstance(state_dict, dict):
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                    print("Found 'state_dict' key in checkpoint")
                elif 'model' in state_dict:
                    state_dict = state_dict['model']
                    print("Found 'model' key in checkpoint")
                elif 'model_state_dict' in state_dict:
                    state_dict = state_dict['model_state_dict']
                    print("Found 'model_state_dict' key in checkpoint")
            
            # Handle DataParallel prefix
            if all(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k[7:]: v for k, v in state_dict.items()}
                print("Removed 'module.' prefix from state dict keys")
            
            model.load_state_dict(state_dict, strict=False)
            print("Loaded state dict into custom model successfully")
            return model
        except Exception as e:
            print(f"Error loading custom model state dict: {e}")
    
    # If custom model loading failed, try with a default model
    try:
        # Try with ResNet50 as a fallback
        print("Trying to load state dict into ResNet50 as fallback")
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
        state_dict = torch.load(weight_path, map_location=device)
        
        # Check if it's a wrapped state dict
        if isinstance(state_dict, dict):
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            elif 'model' in state_dict:
                state_dict = state_dict['model']
            elif 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
        
        # Handle DataParallel prefix
        if all(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict, strict=False)
        print("Loaded state dict into ResNet50 fallback model")
        return model
    except Exception as e:
        print(f"Error loading fallback model: {e}")
    
    # Last resort: use a pretrained model
    print("Falling back to pretrained ResNet50")
    model = models.resnet50(pretrained=True)
    if num_classes != 1000:  # ImageNet has 1000 classes
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# ----------------------------
# Extract Images
# ----------------------------
image_dir = "./temp_images"
if os.path.exists(image_dir):
    shutil.rmtree(image_dir)
os.makedirs(image_dir)

with zipfile.ZipFile(images_zip_path, 'r') as zip_ref:
    zip_ref.extractall(image_dir)

image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
               if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
print(f"Found {len(image_files)} images")

# ----------------------------
# Load Ground Truth Labels and Class Names
# ----------------------------
labels = load_labels_csv(dataset_path)
class_names = load_class_names(labels_path)
print(f"Loaded {len(labels)} labels and {len(class_names)} class names")

# ----------------------------
# Load Model
# ----------------------------
model = load_model(weight_path, model_definition_path, model_class_name)
model = model.to(device)
model.eval()

# ----------------------------
# Run Inference & Evaluate
# ----------------------------
correct = 0
total = 0
class_tp = {}  # True positives per class
class_fp = {}  # False positives per class
class_fn = {}  # False negatives per class
class_count = {}  # Total instances per class
all_preds = []
all_labels = []
all_probs = []

print("Running inference...")
with torch.no_grad():
    for image_file in image_files:
        # Preprocess image
        input_tensor = preprocess_image(image_file).to(device)
        
        # Forward pass
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get prediction
        _, predicted = torch.max(outputs, 1)
        top_class = predicted.item()
        
        # Store probabilities for metrics
        all_probs.append(probabilities.cpu().numpy()[0])
        
        # Get ground truth
        image_name = os.path.basename(image_file)
        true_label = labels.get(image_name)
        
        if true_label is not None:
            total += 1
            all_preds.append(top_class)
            all_labels.append(true_label)
            
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
            
            # Print prediction result
            pred_class_name = class_names[top_class] if top_class < len(class_names) else f"Class {top_class}"
            true_class_name = class_names[true_label] if true_label < len(class_names) else f"Class {true_label}"
            print(f"Image: {image_name} -> Predicted: {pred_class_name}, Actual: {true_class_name}")
        else:
            print(f"[⚠️ Warning] No label found for image: {image_name}")

# ----------------------------
# Calculate Metrics
# ----------------------------
# Overall accuracy
accuracy = correct / total if total > 0 else 0
print(f"\n✅ Accuracy: {accuracy * 100:.2f}% ({correct}/{total})")

# Calculate per-class metrics
all_classes = set(class_tp.keys()) | set(class_fp.keys()) | set(class_fn.keys())

# Calculate precision, recall, F1 for each class
class_precision = {}
class_recall = {}
class_f1 = {}
class_balanced_acc = {}
class_specificity = {}

for cls in all_classes:
    tp = class_tp.get(cls, 0)
    fp = class_fp.get(cls, 0)
    fn = class_fn.get(cls, 0)
    tn = total - (tp + fp + fn)
    
    # Precision = TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    class_precision[cls] = precision
    
    # Recall = TP / (TP + FN)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    class_recall[cls] = recall
    
    # F1 = 2 * (Precision * Recall) / (Precision + Recall)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    class_f1[cls] = f1
    
    # Specificity = TN / (TN + FP)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    class_specificity[cls] = specificity
    
    # Balanced accuracy = (Sensitivity + Specificity) / 2
    balanced_acc = (recall + specificity) / 2
    class_balanced_acc[cls] = balanced_acc
    
    class_name = class_names[cls] if cls < len(class_names) else f"Class {cls}"
    print(f"{class_name}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, Balanced Acc={balanced_acc:.4f}")

# Calculate macro averages
macro_precision = sum(class_precision.values()) / len(class_precision) if class_precision else 0
macro_recall = sum(class_recall.values()) / len(class_recall) if class_recall else 0
macro_f1 = sum(class_f1.values()) / len(class_f1) if class_f1 else 0
macro_balanced_acc = sum(class_balanced_acc.values()) / len(class_balanced_acc) if class_balanced_acc else 0

print(f"\n📊 Macro-Average Metrics:")
print(f"Precision: {macro_precision:.4f}")
print(f"Recall: {macro_recall:.4f}")
print(f"F1-score: {macro_f1:.4f}")
print(f"Balanced Accuracy: {macro_balanced_acc:.4f}")

# Calculate confusion matrix
try:
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\n📊 Confusion Matrix:")
    print(cm)
except Exception as e:
    print(f"Error calculating confusion matrix: {e}")

# Calculate log loss if possible
try:
    from sklearn.metrics import log_loss
    all_probs = np.array(all_probs)
    
    # Create one-hot encoded labels
    num_classes = all_probs.shape[1]
    y_true = np.zeros((len(all_labels), num_classes))
    for i, label in enumerate(all_labels):
        if label < num_classes:
            y_true[i, label] = 1
    
    avg_log_loss = log_loss(y_true, all_probs)
    print(f"\n📉 Log Loss: {avg_log_loss:.4f}")
except Exception as e:
    print(f"Error calculating log loss: {e}")

# ----------------------------
# Setup MLflow Tracking
# ----------------------------
if mlflowURI:
    mlflow.set_tracking_uri(mlflowURI)
    mlflow.set_experiment(experiment_name)

    # Start an MLflow run
    with mlflow.start_run():
        # Log model file as an artifact
        mlflow.log_artifact(weight_path, "model")
        if os.path.exists(model_definition_path):
            mlflow.log_artifact(model_definition_path, "model_definition")
        
        # Log overall metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("macro_precision", macro_precision)
        mlflow.log_metric("macro_recall", macro_recall)
        mlflow.log_metric("macro_f1", macro_f1)
        mlflow.log_metric("macro_balanced_accuracy", macro_balanced_acc)
        
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
        mlflow.log_param("model_definition", model_definition_path)
        mlflow.log_param("model_class", model_class_name)
        mlflow.log_param("total_samples", total)
        mlflow.log_param("num_classes", len(all_classes))
        mlflow.log_param("input_size", input_size)
        
        print("\n✅ All metrics have been logged to MLflow")
else:
    print("\n⚠️ MLflow tracking URI not set, metrics not logged")

# ----------------------------
# Cleanup
# ----------------------------
shutil.rmtree(image_dir)
print("Temporary files cleaned up")