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
from einops import rearrange

# Load variables from .env file
load_dotenv()

mlflowURI = os.getenv("MLFLOW_TRACKING_URI")
weight_path = os.getenv("MODEL_WEIGHTS_PATH")
images_zip_path = os.getenv("IMAGES_ZIP_PATH")
masks_zip_path = os.getenv("MASKS_ZIP_PATH")
dataset_path = os.getenv("DATASET_PATH")
labels_path = os.getenv("DATA_LABELS_PATH")
experiment_name = os.getenv("EXPERIMENT_NAME")
vit_model_size = os.getenv("VIT_MODEL_SIZE", "base")  # base, small, large
num_classes = int(os.getenv("NUM_CLASSES", "21"))  # Default: 21 classes (20 + background)
patch_size = int(os.getenv("PATCH_SIZE", "16"))  # Default patch size for ViT

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------------
# Helper Functions
# ----------------------------
def preprocess_image(image_path, input_size=(512, 512)):
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

def load_mask(mask_path, input_size=(512, 512)):
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

def get_vit_model_name(model_size="base"):
    """Get the correct ViT model name based on size"""
    if model_size.lower() == "small":
        return "vit_small_patch16_384"
    elif model_size.lower() == "large":
        return "vit_large_patch16_384"
    else:  # Default to base
        return "vit_base_patch16_384"

# ----------------------------
# Define Vision Transformer for Segmentation
# ----------------------------
class ViTSegmentation(nn.Module):
    def __init__(self, model_size="base", num_classes=21, patch_size=16):
        super(ViTSegmentation, self).__init__()
        
        # Load ViT as encoder
        vit_model_name = get_vit_model_name(model_size)
        self.encoder = timm.create_model(
            vit_model_name, 
            pretrained=True,
            num_classes=0  # Remove classification head
        )
        
        # Get embedding dimension
        self.embed_dim = self.encoder.embed_dim
        
        # Determine input size and number of patches
        if model_size.lower() == "small":
            self.input_size = 384
        elif model_size.lower() == "large":
            self.input_size = 384
        else:  # base
            self.input_size = 384
        
        self.patch_size = patch_size
        self.num_patches = (self.input_size // patch_size) ** 2
        
        # Decoder (upsampling path)
        self.decoder = nn.Sequential(
            nn.Linear(self.embed_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, patch_size * patch_size * num_classes)
        )
        
        # Final upsampling to original image size
        self.upsample = nn.Upsample(
            scale_factor=patch_size, 
            mode='bilinear', 
            align_corners=True
        )
    
    def forward(self, x):
        # Resize input to expected size if needed
        if x.size(2) != self.input_size or x.size(3) != self.input_size:
            x = nn.functional.interpolate(
                x, 
                size=(self.input_size, self.input_size), 
                mode='bilinear', 
                align_corners=True
            )
        
        # Get patch embeddings from ViT encoder
        x = self.encoder.forward_features(x)  # [B, num_patches+1, embed_dim]
        
        # Remove class token
        x = x[:, 1:, :]  # [B, num_patches, embed_dim]
        
        # Apply decoder to each patch embedding
        x = self.decoder(x)  # [B, num_patches, patch_size*patch_size*num_classes]
        
        # Reshape to form patch-wise segmentation maps
        B = x.size(0)
        x = x.reshape(B, self.num_patches, self.patch_size, self.patch_size, num_classes)
        x = rearrange(
            x, 
            'b (h w) p1 p2 c -> b c (h p1) (w p2)', 
            h=self.input_size//self.patch_size
        )
        
        # Resize to original input size if needed
        if x.size(2) != self.input_size or x.size(3) != self.input_size:
            x = self.upsample(x)
        
        return {'out': x}

# ----------------------------
# Load PyTorch Model
# ----------------------------
def load_model(weight_path, model_size="base", num_classes=21, patch_size=16):
    """Load Vision Transformer segmentation model from .pth file"""
    print(f"Loading ViT-{model_size} segmentation model from: {weight_path}")
    
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
        model = ViTSegmentation(model_size=model_size, num_classes=num_classes, patch_size=patch_size)
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
    print(f"Falling back to fresh ViT-{model_size} segmentation model")
    return ViTSegmentation(model_size=model_size, num_classes=num_classes, patch_size=patch_size)

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
model = load_model(weight_path, vit_model_size, num_classes, patch_size)
model.to(device)
model.eval()

# ----------------------------
# MLFlow Setup
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
        
        # Resize output to match original mask size
        if output.size(2) != resized_size[1] or output.size(3) != resized_size[0]:
            output = nn.functional.interpolate(
                output, 
                size=resized_size[::-1],  # (height, width)
                mode='bilinear', 
                align_corners=True
            )
        
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
print(f"Mean Dice: {mean_dice:.4f}")
print(f"Mean Pixel Accuracy: {mean_pixel_acc:.4f}")
