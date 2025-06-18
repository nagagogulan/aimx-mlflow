import os
import csv
import zipfile
import importlib.util
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import mlflow
import mlflow.tensorflow
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
model_definition_path = os.getenv("MODEL_DEFINITION_PATH", "./model_definition.py")
model_class_name = os.getenv("MODEL_CLASS_NAME", "CustomSegmentationModel")
images_zip_path = os.getenv("IMAGES_ZIP_PATH")
masks_zip_path = os.getenv("MASKS_ZIP_PATH")
dataset_path = os.getenv("DATASET_PATH")
labels_path = os.getenv("DATA_LABELS_PATH")
experiment_name = os.getenv("EXPERIMENT_NAME")
num_classes = int(os.getenv("NUM_CLASSES", "21"))  # Default: 21 classes (20 + background)
input_size = int(os.getenv("INPUT_SIZE", "224"))  # Default input size

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
def preprocess_image(image_path, input_size=(224, 224)):
    """Preprocess image for TensorFlow segmentation model"""
    img = Image.open(image_path).convert("RGB")
    
    # Store original size
    original_size = img.size
    
    # Resize to model input size
    img_resized = img.resize(input_size, Image.BILINEAR)
    
    # Convert to numpy array and normalize
    img_np = np.array(img_resized).astype(np.float32) / 255.0
    
    # Add batch dimension
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

def load_custom_model_class(model_definition_path, model_class_name):
    """Load custom model class from Python file"""
    try:
        print(f"Loading custom model class '{model_class_name}' from {model_definition_path}")
        
        # Load the module
        spec = importlib.util.spec_from_file_location("custom_model_module", model_definition_path)
        custom_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_module)
        
        # Get the model class
        model_class = getattr(custom_module, model_class_name)
        print(f"Successfully loaded custom model class: {model_class}")
        
        return model_class
    except Exception as e:
        print(f"Error loading custom model class: {e}")
        return None

# ----------------------------
# Create Default Segmentation Model
# ----------------------------
def create_default_segmentation_model(input_size=224, num_classes=21):
    """Create a default U-Net style segmentation model"""
    inputs = tf.keras.Input(shape=(input_size, input_size, 3))
    
    # Encoder
    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    skip1 = x
    x = tf.keras.layers.MaxPooling2D()(x)
    
    x = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    skip2 = x
    x = tf.keras.layers.MaxPooling2D()(x)
    
    x = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    skip3 = x
    x = tf.keras.layers.MaxPooling2D()(x)
    
    x = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(512, 3, padding='same', activation='relu')(x)
    
    # Decoder
    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.Concatenate()([x, skip3])
    x = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x)
    
    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.Concatenate()([x, skip2])
    x = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x)
    
    x = tf.keras.layers.UpSampling2D()(x)
    x = tf.keras.layers.Concatenate()([x, skip1])
    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    
    # Output
    outputs = tf.keras.layers.Conv2D(num_classes, 1, padding='same', activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# ----------------------------
# Load TensorFlow Model
# ----------------------------
def load_model_with_custom_objects():
    """Load segmentation model from .h5 file with custom objects if needed"""
    print(f"Loading segmentation model from: {weight_path}")
    
    try:
        # Try loading as a complete model
        model = tf.keras.models.load_model(weight_path)
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
            # Create an instance of the custom model
            custom_model = custom_model_class(
                input_shape=(input_size, input_size, 3),
                num_classes=num_classes
            )
            
            # Build the model
            custom_model.build((None, input_size, input_size, 3))
            
            # Load weights
            custom_model.load_weights(weight_path)
            print(f"Loaded weights into custom model: {model_class_name}")
            
            return custom_model
        except Exception as e:
            print(f"Error loading custom model: {e}")
    
    # Try loading with custom objects scope
    try:
        with tf.keras.utils.custom_object_scope({'CustomSegmentationModel': custom_model_class}):
            model = tf.keras.models.load_model(weight_path)
            print("Loaded model with custom object scope")
            return model
    except Exception as e:
        print(f"Error loading with custom object scope: {e}")
    
    # Create a default model and try to load weights
    try:
        print("Creating default segmentation model and loading weights")
        model = create_default_segmentation_model(input_size=input_size, num_classes=num_classes)
        model.load_weights(weight_path)
        print("Loaded weights into default model")
        return model
    except Exception as e:
        print(f"Error loading weights into default model: {e}")
        
        # Return default model with random weights as last resort
        print("Using default model with random weights")
        return create_default_segmentation_model(input_size=input_size, num_classes=num_classes)

# ----------------------------
# MLFlow Setup
# ----------------------------
mlflow.set_tracking_uri(mlflowURI)
mlflow.set_experiment(experiment_name)

# Load the model
model = load_model_with_custom_objects()
model.summary()

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
    img_np, original_img, original_size = preprocess_image(image_path, input_size=(input_size, input_size))
    
    # Run inference
    prediction = model.predict(img_np)
    
    # Process output based on shape
    if len(prediction.shape) == 4:  # [N, H, W, C]
        # Get class predictions
        pred_mask = np.argmax(prediction[0], axis=-1)
    else:
        print(f"Unexpected output shape: {prediction.shape}")
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
# Calculate mean IoU for each class
mean_ious = {class_idx: np.mean(ious) if ious else 0.0 
             for class_idx, ious in class_ious.items()}

# Calculate mean Dice for each class
mean_dices = {class_idx: np.mean(dices) if dices else 0.0
              for class_idx, dices in class_dices.items()}

# Calculate pixel accuracy
mean_pixel_accuracy = np.mean(pixel_accuracies)

# Log metrics to MLFlow
with mlflow.start_run():
    mlflow.log_param("num_classes", num_classes)
    mlflow.log_param("input_size", input_size)
    mlflow.log_metric("mean_pixel_accuracy", mean_pixel_accuracy)
    
    for class_idx in range(num_classes):
        mlflow.log_metric(f"mean_iou_class_{class_idx}", mean_ious[class_idx])
        mlflow.log_metric(f"mean_dice_class_{class_idx}", mean_dices[class_idx])

    print("MLFlow metrics logged successfully.")
