import os
import csv
import zipfile
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import mlflow
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import shutil
import matplotlib.pyplot as plt
import cv2
import timm
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

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------------
# Helper Functions
# ----------------------------
def preprocess_image(image_path, input_size=(520, 520)):
    """Preprocess image for segmentation model"""
    img = Image.open(image_path).convert("RGB")
    
    # Resize while maintaining aspect ratio
    img_resized = img.copy()
    if input_size:
        img_resized = img.resize(input_size, Image.BILINEAR)
    
    # Apply transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    img_tensor = transform(img_resized).unsqueeze(0)
    return img_tensor, img, img_resized.size

def load_mask(mask_path, input_size=(520, 520)):
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
# Define EfficientNet-B4 Segmentation Model
# ----------------------------
class EfficientNetB4Segmentation(nn.Module):
    def __init__(self, num_classes=21):
        super(EfficientNetB4Segmentation, self).__init__()
        
        # Load EfficientNet-B4 as encoder (backbone)
        self.encoder = timm.create_model('efficientnet_b4', pretrained=True, features_only=True)
        
        # Get feature dimensions from the encoder
        feature_dims = self.encoder.feature_info.channels()
        
        # Create decoder (upsampling path)
        self.decoder_stage1 = self._make_decoder_stage(feature_dims[-1], feature_dims[-2], feature_dims[-2])
        self.decoder_stage2 = self._make_decoder_stage(feature_dims[-2], feature_dims[-3], feature_dims[-3])
        self.decoder_stage3 = self._make_decoder_stage(feature_dims[-3], feature_dims[-4], feature_dims[-4])
        self.decoder_stage4 = self._make_decoder_stage(feature_dims[-4], feature_dims[-5], feature_dims[-5])
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Conv2d(feature_dims[-5], 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
        
        # Upsampling to original image size
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
    
    def _make_decoder_stage(self, in_channels, skip_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
    
    def forward(self, x):
        # Get encoder features
        features = self.encoder(x)
        
        # Decoder path with skip connections
        x = self.decoder_stage1(torch.cat([features[-1], features[-2]], dim=1))
        x = self.decoder_stage2(torch.cat([x, features[-3]], dim=1))
        x = self.decoder_stage3(torch.cat([x, features[-4]], dim=1))
        x = self.decoder_stage4(torch.cat([x, features[-5]], dim=1))
        
        # Final classification
        x = self.classifier(x)
        
        # Upsample to original image size
        x = self.upsample(x)
        
        return {'out': x}

# ----------------------------
# Load PyTorch Model
# ----------------------------
def load_model(weight_path, num_classes=21):
    """Load EfficientNet-B4 segmentation model from .pth file"""
    print(f"Loading EfficientNet-B4 segmentation model from: {weight_path}")
    
    try:
        # Try loading as a complete model
        model = torch.load(weight_path, map_location=device)
        if isinstance(model, nn.Module):
            print("Loaded complete model")
            return model
    except Exception as e:
        print(f"Not a complete model: {e}")
    
    try:
        # Try loading as a state dict
        model = EfficientNetB4Segmentation(num_classes=num_classes)
        state_dict = torch.load(weight_path, map_location=device)
        
        # Check if it's a wrapped state dict (common in checkpoints)
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
        print("Loaded state dict successfully")
        return model
    except Exception as e:
        print(f"Error loading state dict: {e}")
    
    # Fallback to fresh model
    print("Falling back to fresh EfficientNet-B4 segmentation model")
    return EfficientNetB4Segmentation(num_classes=num_classes)

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
# Load Model
# ----------------------------
model = load_model(weight_path, num_classes)
model.to(device)
model.eval()

# ----------------------------
# Set MLflow Tracking URI and Experiment
# ----------------------------
mlflow.set_tracking_uri(mlflowURI)
mlflow.set_experiment(experiment_name)

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
with torch.no_grad():
    for idx, (image_path, mask_path) in enumerate(tqdm(image_mask_pairs)):
        # Load and preprocess image
        img_tensor, original_img, resized_size = preprocess_image(image_path)
        img_tensor = img_tensor.to(device)
        
        # Run inference
        output = model(img_tensor)
        
        # Extract output
        output = output['out']
        
        # Get predicted mask
        pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        
        # Load ground truth mask
        gt_mask = load_mask(mask_path, input_size=resized_size)
        
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
            original_np = np.array(original_img.resize(resized_size))
            
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
print(f"Mean Dice Coefficient: {mean_dice:.4f}")
print(f"Mean Pixel Accuracy: {mean_pixel_acc:.4f}")

# Per-class metrics
print("\n===== Per-Class Metrics =====")
for class_idx in valid_classes:
    if class_idx < len(class_names):
        class_name = class_names[class_idx]
    else:
        class_name = f"Class {class_idx}"
    
    miou = class_miou[class_idx]
    mdice = class_mdice[class_idx]
    
    print(f"{class_name}: IoU={miou:.4f}, Dice={mdice:.4f}")

# ----------------------------
# Plot Class-wise Metrics
# ----------------------------
# Filter out classes with no instances
valid_class_names = [class_names[idx] if idx < len(class_names) else f"Class {idx}" for idx in valid_classes]
valid_mious = [class_miou[idx] for idx in valid_classes]
valid_mdices = [class_mdice[idx] for idx in valid_classes]

# Plot IoU and Dice per class
if valid_classes:
    plt.figure(figsize=(12, 6))
    x = np.arange(len(valid_classes))
    width = 0.35

    plt.bar(x - width/2, valid_mious, width, label='IoU')
    plt.bar(x + width/2, valid_mdices, width, label='Dice')

    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.title('IoU and Dice Coefficient per Class')

