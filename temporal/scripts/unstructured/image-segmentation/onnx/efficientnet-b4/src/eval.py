import os
import csv
import zipfile
import numpy as np
import onnxruntime as ort
from PIL import Image
import mlflow
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, confusion_matrix
import shutil
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import seaborn as sns

# Load variables from .env file
load_dotenv()

mlflowURI = os.getenv("MLFLOW_TRACKING_URI")
weight_path = os.getenv("MODEL_WEIGHTS_PATH")
images_zip_path = os.getenv("IMAGES_ZIP_PATH")
masks_zip_path = os.getenv("MASKS_ZIP_PATH")
dataset_path = os.getenv("DATASET_PATH")
labels_path = os.getenv("DATA_LABELS_PATH")
experiment_name = os.getenv("EXPERIMENT_NAME")
num_classes = int(os.getenv("NUM_CLASSES", "21"))  # Default: 21 classes (20 + background)
input_size = int(os.getenv("INPUT_SIZE", "380"))  # Default EfficientNet-B4 input size

# ----------------------------
# Helper Functions
# ----------------------------
def preprocess_image(image_path, input_size=(380, 380)):
    """Preprocess image for ONNX segmentation model with EfficientNet-B4 backbone"""
    img = Image.open(image_path).convert("RGB")
    
    # Store original size
    original_size = img.size
    
    # Resize to model input size
    img_resized = img.resize(input_size, Image.BILINEAR)
    
    # Convert to numpy array and normalize
    img_np = np.array(img_resized).astype(np.float32) / 255.0
    
    # Normalize with ImageNet mean and std
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    img_np = (img_np - mean) / std
    
    # Transpose from HWC to NCHW format (batch, channels, height, width)
    img_np = img_np.transpose(2, 0, 1)
    img_np = np.expand_dims(img_np, axis=0)
    
    return img_np, img, original_size

def load_mask(mask_path, input_size=None):
    """Load ground truth segmentation mask"""
    mask = Image.open(mask_path)
    
    # Resize to match model input size
    if input_size:
        mask = mask.resize(input_size, Image.NEAREST)
    
    mask_np = np.array(mask)
    return mask_np

def load_class_names(labels_path):
    """Load class names from file"""
    try:
        with open(labels_path, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]
        return class_names
    except Exception as e:
        print(f"Error loading class names: {e}")
        return [f"Class {i}" for i in range(num_classes)]

def calculate_iou(pred_mask, gt_mask, class_idx):
    """Calculate IoU for a specific class"""
    pred_class = (pred_mask == class_idx).astype(np.uint8)
    gt_class = (gt_mask == class_idx).astype(np.uint8)
    
    intersection = np.logical_and(pred_class, gt_class).sum()
    union = np.logical_or(pred_class, gt_class).sum()
    
    if union == 0:
        return 0.0
    
    return intersection / union

def calculate_dice(pred_mask, gt_mask, class_idx):
    """Calculate Dice coefficient for a specific class"""
    pred_class = (pred_mask == class_idx).astype(np.uint8)
    gt_class = (gt_mask == class_idx).astype(np.uint8)
    
    intersection = np.logical_and(pred_class, gt_class).sum()
    total = pred_class.sum() + gt_class.sum()
    
    if total == 0:
        return 0.0
    
    return 2 * intersection / total

def create_color_map(num_classes):
    """Create a color map for visualization"""
    color_map = np.zeros((num_classes, 3), dtype=np.uint8)
    
    # Set background to black
    color_map[0] = [0, 0, 0]
    
    # Generate random colors for other classes
    np.random.seed(42)
    for i in range(1, num_classes):
        color_map[i] = np.random.randint(0, 255, 3)
    
    return color_map

def apply_color_map(mask, color_map):
    """Apply color map to segmentation mask"""
    height, width = mask.shape
    colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
    
    for class_idx in range(len(color_map)):
        colored_mask[mask == class_idx] = color_map[class_idx]
    
    return colored_mask

# ----------------------------
# Load ONNX Model
# ----------------------------
print(f"Loading ONNX EfficientNet-B4 segmentation model from: {weight_path}")
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
    
    # Determine input size from model
    if len(input_shape) == 4 and input_shape[2] is not None and input_shape[3] is not None:
        input_height = input_shape[2]
        input_width = input_shape[3]
        input_size = (input_width, input_height)
        print(f"Using input size from model: {input_size}")
    else:
        input_size = (input_size, input_size)  # Default from environment variable
        print(f"Using default input size: {input_size}")
    
except Exception as e:
    print(f"Error loading model: {e}")
    # Try again with CPU only
    try:
        session = ort.InferenceSession(weight_path)
        print("Model loaded successfully with CPU")
        input_name = session.get_inputs()[0].name
        input_size = (input_size, input_size)  # Default from environment variable
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise

# ----------------------------
# Extract Images and Masks
# ----------------------------
image_dir = "./temp_images"
mask_dir = "./temp_masks"

if os.path.exists(image_dir):
    shutil.rmtree(image_dir)
if os.path.exists(mask_dir):
    shutil.rmtree(mask_dir)

os.makedirs(image_dir)
os.makedirs(mask_dir)

print(f"Extracting images from: {images_zip_path}")
with zipfile.ZipFile(images_zip_path, 'r') as zip_ref:
    zip_ref.extractall(image_dir)

print(f"Extracting masks from: {masks_zip_path}")
with zipfile.ZipFile(masks_zip_path, 'r') as zip_ref:
    zip_ref.extractall(mask_dir)

# Get image and mask files
image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
               if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
mask_files = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) 
              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Match image and mask files by name
image_mask_pairs = []
for image_file in image_files:
    image_name = os.path.basename(image_file)
    mask_name = image_name  # Assuming mask files have the same name as image files
    
    # Find corresponding mask file
    matching_mask = [m for m in mask_files if os.path.basename(m) == mask_name]
    if matching_mask:
        image_mask_pairs.append((image_file, matching_mask[0]))

print(f"Found {len(image_mask_pairs)} image-mask pairs")

# ----------------------------
# Load Class Names
# ----------------------------
class_names = load_class_names(labels_path)
print(f"Loaded {len(class_names)} class names")

# Create color map for visualization
color_map = create_color_map(num_classes)

# ----------------------------
# Run Inference & Evaluate
# ----------------------------
# Metrics storage
class_ious = {i: [] for i in range(num_classes)}
class_dices = {i: [] for i in range(num_classes)}
pixel_accuracies = []

# Create output directory for visualizations
output_dir = "./output_visualizations"
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir)

print("Running inference...")
for idx, (image_path, mask_path) in enumerate(tqdm(image_mask_pairs)):
    # Load and preprocess image
    img_np, original_img, original_size = preprocess_image(image_path, input_size=input_size)
    
    # Run inference
    outputs = session.run(None, {input_name: img_np})
    output = outputs[0]  # Assuming the first output is the segmentation map
    
    # Process output based on shape
    if len(output.shape) == 4:  # [N, C, H, W]
        # Get class predictions
        pred_mask = np.argmax(output[0], axis=0)
    else:
        print(f"Unexpected output shape: {output.shape}")
        continue
    
    # Resize prediction to original image size
    pred_mask_pil = Image.fromarray(pred_mask.astype(np.uint8))
    pred_mask_resized = pred_mask_pil.resize(original_size, Image.NEAREST)
    pred_mask = np.array(pred_mask_resized)
    
    # Load ground truth mask
    gt_mask = load_mask(mask_path, input_size=original_size)
    
    # Calculate metrics
    for class_idx in range(num_classes):
        if class_idx in np.unique(gt_mask):
            iou = calculate_iou(pred_mask, gt_mask, class_idx)
            dice = calculate_dice(pred_mask, gt_mask, class_idx)
            class_ious[class_idx].append(iou)
            class_dices[class_idx].append(dice)
    
    # Calculate pixel accuracy
    pixel_acc = np.mean(pred_mask == gt_mask)
    pixel_accuracies.append(pixel_acc)
    
    # Visualize results
    if idx < 10:  # Limit visualizations to first 10 images
        # Apply color map to masks
        colored_pred = apply_color_map(pred_mask, color_map)
        colored_gt = apply_color_map(gt_mask, color_map)
        
        # Convert original image to numpy array
        original_np = np.array(original_img)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(original_np)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        axes[1].imshow(colored_gt)
        axes[1].set_title("Ground Truth")
        axes[1].axis('off')
        
        axes[2].imshow(colored_pred)
        axes[2].set_title("Prediction")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"result_{idx}.png"))
        plt.close()
        
        # Create overlay visualization
        alpha = 0.5
        overlay = cv2.addWeighted(
            original_np, 1 - alpha,
            colored_pred, alpha, 0
        )
        
        plt.figure(figsize=(8, 8))
        plt.imshow(overlay)
        plt.title("Segmentation Overlay")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"overlay_{idx}.png"))
        plt.close()

# ----------------------------
# Calculate Overall Metrics
# ----------------------------
# Mean IoU
class_miou = {class_idx: np.mean(ious) if ious else 0 
              for class_idx, ious in class_ious.items()}
valid_classes = [idx for idx in range(num_classes) if class_miou[idx] > 0]
mean_iou = np.mean([class_miou[idx] for idx in valid_classes]) if valid_classes else 0

# Mean Dice
class_mdice = {class_idx: np.mean(dices) if dices else 0 
               for class_idx, dices in class_dices.items()}
mean_dice = np.mean([class_mdice[idx] for idx in valid_classes]) if valid_classes else 0

# Mean Pixel Accuracy
mean_pixel_acc = np.mean(pixel_accuracies) if pixel_accuracies else 0

print("\n===== Segmentation Metrics =====")
print(f"Mean IoU: {mean_iou:.4f}")
print(f"Mean Dice: {mean_dice:.4f}")
print(f"Mean Pixel Accuracy: {mean_pixel_acc:.4f}")

# Per-class metrics
print("\n===== Per-Class Metrics =====")
for class_idx in valid_classes:
    if class_idx < len(class_names):
        class_name = class_names[class_idx]
    else:
        class_name = f"Class {class_idx}"
    
    print(f"{class_name}:")
    print(f"  IoU: {class_miou[class_idx]:.4f}")
    print(f"  Dice: {class_mdice[class_idx]:.4f}")

# ----------------------------
# Create Confusion Matrix
# ----------------------------
if len(image_mask_pairs) > 0:
    # Create a confusion matrix for the most common classes
    # Flatten all predictions and ground truths
    all_preds = []
    all_gts = []
    
    for idx, (image_path, mask_path) in enumerate(image_mask_pairs):
        # Load and preprocess image
        img_np, _, original_size = preprocess_image(image_path, input_size=input_size)
        
        # Run inference
        outputs = session.run(None, {input_name: img_np})
        output = outputs[0]
        
        # Process output
        if len(output.shape) == 4:  # [N, C, H, W]
            pred_mask = np.argmax(output[0], axis=0)
        else:
            continue
        
        # Resize prediction
        pred_mask_pil = Image.fromarray(pred_mask.astype(np.uint8))
        pred_mask_resized = pred_mask_pil.resize(original_size, Image.NEAREST)
        pred_mask = np.array(pred_mask_resized)
        
        # Load ground truth mask
        gt_mask = load_mask(mask_path, input_size=original_size)
        
        # Flatten and append
        all_preds.extend(pred_mask.flatten())
        all_gts.extend(gt_mask.flatten())
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_gts = np.array(all_gts)
    
    # Get unique classes in ground truth
    unique_classes = np.unique(all_gts)
    
    # Create confusion matrix for classes that actually appear
    cm = confusion_matrix(all_gts, all_preds, labels=unique_classes)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[class_names[i] if i < len(class_names) else f"Class {i}" for i in unique_classes],
                yticklabels=[class_names[i] if i < len(class_names) else f"Class {i}" for i in unique_classes])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

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
    mlflow.log_metric("mean_iou", mean_iou)
    mlflow.log_metric("mean_dice", mean_dice)
    mlflow.log_metric("mean_pixel_accuracy", mean_pixel_acc)
    
    # Log per-class metrics
    for class_idx in valid_classes:
        if class_idx < len(class_names):
            class