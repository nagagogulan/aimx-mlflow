import os
import csv
import zipfile
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import mlflow
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, log_loss
import shutil

# Load variables from .env file
load_dotenv()

mlflowURI = os.getenv("MLFLOW_TRACKING_URI")
weight_path = os.getenv("MODEL_WEIGHTS_PATH")
images_zip_path = os.getenv("IMAGES_ZIP_PATH")
dataset_path = os.getenv("DATASET_PATH")
labels_path = os.getenv("DATA_LABELS_PATH")
experiment_name = os.getenv("EXPERIMENT_NAME")
input_size = int(os.getenv("INPUT_SIZE", "380"))  # Default EfficientNet-B4 input size

# Set memory growth for GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"Error setting memory growth: {e}")
else:
    print("Using CPU for inference")

# ----------------------------
# Helper Functions
# ----------------------------
def preprocess_image(image_path, input_size=(380, 380)):
    """Preprocess image for EfficientNet-B4"""
    img = load_img(image_path, target_size=input_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    # EfficientNet preprocessing: scale to [0, 1]
    return img_array / 255.0

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

# ----------------------------
# Load TensorFlow Model
# ----------------------------
print(f"Loading EfficientNet-B4 model from: {weight_path}")
try:
    # Custom objects might be needed for some models
    custom_objects = {}
    model = load_model(weight_path, custom_objects=custom_objects)
    model.summary()
    
    # Check if the model has the expected EfficientNet architecture
    if 'efficient' not in weight_path.lower() and not any('efficient' in layer.name.lower() for layer in model.layers):
        print("Warning: This may not be an EfficientNet model. Continuing anyway...")
    
    # Get model input shape
    input_shape = model.input_shape
    if input_shape and len(input_shape) == 4:
        if input_shape[1] is not None and input_shape[2] is not None:
            model_input_size = (input_shape[1], input_shape[2])
            if model_input_size != (input_size, input_size):
                print(f"Warning: Input size from model ({model_input_size}) differs from config ({input_size})")
                print(f"Using model's input size: {model_input_size}")
                input_size = model_input_size[0]  # Assuming square input
    
except Exception as e:
    print(f"Error loading model: {e}")
    print("Attempting to load as an EfficientNet-B4 model from TensorFlow...")
    try:
        # Try loading as a standard EfficientNet-B4 model
        import tensorflow_hub as hub
        model = hub.load("https://tfhub.dev/tensorflow/efficientnet/b4/classification/1")
        print("Loaded default EfficientNet-B4 model with ImageNet weights")
        
        # Create a wrapper function for prediction to match the Keras API
        def predict_wrapper(images):
            return model(images)
        
        model.predict = predict_wrapper
    except Exception as e2:
        print(f"Failed to load fallback model: {e2}")
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
    input_data = preprocess_image(image_file, input_size=(input_size, input_size))
    
    # Run inference
    predictions = model.predict(input_data, verbose=0)
    
    # Get top prediction
    top_class = int(np.argmax(predictions[0]))
    
    # Store probabilities for metrics
    all_probs.append(predictions[0])
    
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
        
        # Log TensorFlow model directly
        try:
            mlflow.tensorflow.log_model(model, "tf_model")
        except Exception as e:
            print(f"Warning: Could not log model directly to MLflow: {e}")
        
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
        mlflow.log_param("model_type", "EfficientNet-B4-TensorFlow")
        mlflow.log_param("input_size", input_size)
        
        print("\n✅ All metrics have been logged to MLflow")
else:
    print("\n⚠️ MLflow tracking URI not set, metrics not logged")

# ----------------------------
# Cleanup
# ----------------------------
shutil.rmtree(image_dir)
print("Temporary files cleaned up")