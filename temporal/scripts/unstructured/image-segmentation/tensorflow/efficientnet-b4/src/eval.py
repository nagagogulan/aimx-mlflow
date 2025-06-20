import os
import csv
import zipfile
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
images_zip_path = os.getenv("IMAGES_ZIP_PATH")
masks_zip_path = os.getenv("MASKS_ZIP_PATH")
dataset_path = os.getenv("DATASET_PATH")
labels_path = os.getenv("DATA_LABELS_PATH")
experiment_name = os.getenv("EXPERIMENT_NAME")
num_classes = int(os.getenv("NUM_CLASSES", "21"))  # Default: 21 classes (20 + background)
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
    """Preprocess image for TensorFlow segmentation model with EfficientNet-B4 backbone"""
    img = Image.open(image_path).convert("RGB")
    
    # Store original size
    original_size = img.size
    
    # Resize to model input size
    img_resized = img.resize(input_size, Image.BILINEAR)
    
    # Convert to numpy array and normalize
    img_np = np.array(img_resized).astype(np.float32) / 255.0
    
    # Normalize with ImageNet mean and std
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = (img_np - mean) / std
    
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

# ----------------------------
# Create EfficientNet-B4 Segmentation Model
# ----------------------------
def create_efficientnet_b4_segmentation_model(input_size=380, num_classes=21):
    """Create EfficientNet-B4 segmentation model"""
    try:
        # Try to import EfficientNet from TensorFlow Hub
        import tensorflow_hub as hub
        
        # Load EfficientNet-B4 as feature extractor
        base_model_url = "https://tfhub.dev/tensorflow/efficientnet/b4/feature-vector/1"
        base_model = hub.KerasLayer(base_model_url, trainable=False)
        
        # Create model
        inputs = tf.keras.Input(shape=(input_size, input_size, 3))
        x = base_model(inputs)
        
        # Reshape to spatial features
        x = tf.keras.layers.Reshape((12, 12, 1792))(x)  # EfficientNet-B4 output shape
        
        # Upsampling path
        x = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x)
        x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)  # 24x24
        
        x = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x)
        x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)  # 48x48
        
        x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
        x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)  # 96x96
        
        x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(x)
        x = tf.keras.layers.UpSampling2D(size=(4, 4))(x)  # 384x384 (close to 380x380)
        
        # Final output
        outputs = tf.keras.layers.Conv2D(num_classes, 1, padding='same', activation='softmax')(x)
        
        # Create model
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        return model
    
    except Exception as e:
        print(f"Error creating EfficientNet-B4 model: {e}")
        
        # Fallback to standard EfficientNet implementation
        try:
            from efficientnet.tfkeras import EfficientNetB4
            
            # Create model with EfficientNet-B4 backbone
            base_model = EfficientNetB4(
                weights='imagenet',
                include_top=False,
                input_shape=(input_size, input_size, 3)
            )
            
            # Create encoder-decoder architecture
            inputs = base_model.input
            x = base_model.output
            
            # Upsampling path
            x = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x)
            x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
            
            x = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x)
            x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
            
            x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
            x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
            
            x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(x)
            x = tf.keras.layers.UpSampling2D(size=(4, 4))(x)
            
            # Final output
            outputs = tf.keras.layers.Conv2D(num_classes, 1, padding='same', activation='softmax')(x)
            
            # Create model
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            
            return model
            
        except Exception as e2:
            print(f"Error creating fallback EfficientNet-B4 model: {e2}")
            
            # Final fallback to standard Keras applications
            base_model = tf.keras.applications.MobileNetV2(
                input_shape=(input_size, input_size, 3),
                include_top=False,
                weights='imagenet'
            )
            
            # Create encoder-decoder architecture
            inputs = base_model.input
            x = base_model.output
            
            # Upsampling path
            x = tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu')(x)
            x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
            
            x = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')(x)
            x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
            
            x = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')(x)
            x = tf.keras.layers.UpSampling2D(size=(2, 2))(x)
            
            x = tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu')(x)
            x = tf.keras.layers.UpSampling2D(size=(4, 4))(x)
            
            # Final output
            outputs = tf.keras.layers.Conv2D(num_classes, 1, padding='same', activation='softmax')(x)
            
            # Create model
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            
            print("Created fallback MobileNetV2 model instead of EfficientNet-B4")
            return model

# ----------------------------
# Load TensorFlow Model
# ----------------------------
print(f"Loading EfficientNet-B4 segmentation model from: {weight_path}")

try:
    # Try loading the model directly
    custom_objects = None
    model = load_model(weight_path, custom_objects=custom_objects)
    print("Model loaded successfully")
    
    # Print model summary
    model.summary()
    
except Exception as e:
    print(f"Error loading model: {e}")
    print("Attempting to create EfficientNet-B4 segmentation model...")
    
    try:
        # Create EfficientNet-B4 segmentation model
        model = create_efficientnet_b4_segmentation_model(input_size=input_size, num_classes=num_classes)
        
        # Load weights
        model.load_weights(weight_path)
        print("Loaded weights into EfficientNet-B4 segmentation model")
        
        # Print model summary
        model.summary()
        
    except Exception as e2:
        print(f"Failed to load weights: {e2}")
        print("Using model with ImageNet weights")

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
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"visualization_{idx}.png"))
        plt.close(fig)

# ----------------------------
# MLFlow Integration
# ----------------------------
# Log metrics to MLFlow
mlflow.set_tracking_uri(mlflowURI)
mlflow.set_experiment(experiment_name)

with mlflow.start_run():
    mlflow.log_param("model_weights", weight_path)
    mlflow.log_param("input_size", input_size)
    mlflow.log_param("num_classes", num_classes)
    
    # Log metrics
    avg_pixel_acc = np.mean(pixel_accuracies)
    mlflow.log_metric("avg_pixel_accuracy", avg_pixel_acc)
    
    for class_idx in range(num_classes):
        avg_iou = np.mean(class_ious[class_idx]) if class_ious[class_idx] else 0
        avg_dice = np.mean(class_dices[class_idx]) if class_dices[class_idx] else 0
        
        mlflow.log_metric(f"Class_{class_idx}_IoU", avg_iou)
        mlflow.log_metric(f"Class_{class_idx}_Dice", avg_dice)
    
    print("Logged metrics to MLFlow successfully")
