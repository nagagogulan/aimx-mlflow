import os
import csv
from dotenv import load_dotenv
import onnxruntime as ort
import numpy as np
from PIL import Image
import shutil
import zipfile
import mlflow
from sklearn.metrics import confusion_matrix, log_loss, precision_recall_fscore_support
import json

# Load variables from .env file
load_dotenv()

weight_path = os.getenv("MODEL_WIGHTS_PATH")
images_zip_path = os.getenv("IMAGES_ZIP_PATH")
labels_path = os.getenv("DATA_LABELS_PATH")
dataset_path = os.getenv("DATASET_PATH")
mlflowURI = os.getenv("MLFLOW_TRACKING_URI")
experiment_name = os.getenv("EXPERIMENT_NAME")
input_size = int(os.getenv("INPUT_SIZE", "224"))  # Default input size
model_config_path = os.getenv("MODEL_CONFIG_PATH", "./model_config.json")
num_classes = int(os.getenv("NUM_CLASSES", "1000"))

# ----------------------------
# Helper Functions
# ----------------------------
def preprocess_image(image_path, input_size=(224, 224), config=None):
    """Preprocess image for custom ONNX model"""
    image = Image.open(image_path).convert("RGB")
    image = image.resize(input_size)
    
    # Convert to numpy array
    image_data = np.array(image).astype(np.float32)
    
    # Apply normalization based on config or use default ImageNet normalization
    if config and 'preprocessing' in config:
        # Use custom preprocessing from config
        if config['preprocessing']['type'] == 'normalize':
            mean = np.array(config['preprocessing'].get('mean', [0.485, 0.456, 0.406]))
            std = np.array(config['preprocessing'].get('std', [0.229, 0.224, 0.225]))
            image_data = image_data / 255.0  # Scale to [0,1]
            image_data = (image_data - mean) / std
        elif config['preprocessing']['type'] == 'scale':
            # Simple scaling to [-1,1] or [0,1]
            scale = config['preprocessing'].get('scale', 255.0)
            offset = config['preprocessing'].get('offset', 0.0)
            image_data = (image_data / scale) - offset
    else:
        # Default ImageNet normalization
        image_data = image_data / 255.0  # Scale to [0,1]
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_data = (image_data - mean) / std
    
    # ONNX models typically expect NCHW format
    image_data = np.transpose(image_data, (2, 0, 1))  # Channels first
    image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
    return image_data

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

def load_model_config(config_path):
    """Load model configuration from JSON file if it exists"""
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading model config: {e}")
    return None

# ----------------------------
# Load Model Configuration
# ----------------------------
model_config = load_model_config(model_config_path)
if model_config:
    print(f"Loaded model configuration from: {model_config_path}")
    # Override environment variables with config if available
    if 'input_size' in model_config:
        input_size = model_config['input_size']
        print(f"Using input size from config: {input_size}")
    if 'num_classes' in model_config:
        num_classes = model_config['num_classes']
        print(f"Using num_classes from config: {num_classes}")
else:
    print(f"No model configuration found at {model_config_path}, using defaults")

# ----------------------------
# Load ONNX Model
# ----------------------------
print(f"Loading custom ONNX model from: {weight_path}")
try:
    # Set session options for better performance
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # Check if GPU is available
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(weight_path, session_options, providers=providers)
    
    # Fall back to CPU if GPU is not available
    if 'CUDAExecutionProvider' not in session.get_providers():
        print("CUDA not available, using CPU for inference")
    else:
        print("Using CUDA for inference")
        
    print("Model loaded successfully")
    
    # Print model input details
    model_inputs = session.get_inputs()
    input_name = model_inputs[0].name
    input_shape = model_inputs[0].shape
    print(f"Model input name: {input_name}")
    print(f"Model input shape: {input_shape}")
    
    # Try to infer input size from model if not specified
    if input_shape and len(input_shape) == 4:  # NCHW format
        if input_shape[2] is not None and input_shape[3] is not None:
            inferred_size = (input_shape[2], input_shape[3])
            if inferred_size != (input_size, input_size):
                print(f"Warning: Input size from model ({inferred_size}) differs from config ({input_size})")
                print(f"Using model's input size: {inferred_size}")
                input_size = inferred_size[0]  # Assuming square input
    
except Exception as e:
    print(f"Error loading model: {e}")
    # Try again with CPU only
    try:
        session = ort.InferenceSession(weight_path)
        print("Model loaded successfully with CPU")
        input_name = session.get_inputs()[0].name
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise

# ----------------------------
# Extract Images
# ----------------------------
image_dir = "./temp_images"
if os.path.exists(image_dir):
    shutil.rmtree(image_dir)
os.makedirs(image_dir)

print(f"Extracting images from: {images_zip_path}")
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
for image_file in image_files:
    # Preprocess image
    input_data = preprocess_image(image_file, input_size=(input_size, input_size), config=model_config)
    
    # Run inference
    outputs = session.run(None, {input_name: input_data})
    predictions = outputs[0]
    
    # Get softmax probabilities
    exp_preds = np.exp(predictions - np.max(predictions, axis=1, keepdims=True))
    probabilities = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)
    
    # Get top prediction
    top_class = int(np.argmax(predictions))
    
    # Store probabilities for metrics
    all_probs.append(probabilities[0])
    
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
    cm = confusion_matrix(all_labels, all_preds)
    print("\n📊 Confusion Matrix:")
    print(cm)
except Exception as e:
    print(f"Error calculating confusion matrix: {e}")

# Calculate log loss if possible
try:
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
        
        # Log model config if available
        if model_config:
            with open("model_config_used.json", "w") as f:
                json.dump(model_config, f, indent=2)
            mlflow.log_artifact("model_config_used.json")
        
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
        mlflow.log_param("total_samples", total)
        mlflow.log_param("num_classes", len(all_classes))
        mlflow.log_param("model_type", "Custom-ONNX")
        mlflow.log_param("input_size", input_size)
        
        print("\n✅ All metrics have been logged to MLflow")
else:
    print("\n⚠️ MLflow tracking URI not set, metrics not logged")

# ----------------------------
# Cleanup
# ----------------------------
shutil.rmtree(image_dir)
print("Temporary files cleaned up")