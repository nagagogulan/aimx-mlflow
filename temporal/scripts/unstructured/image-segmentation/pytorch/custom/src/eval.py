import os
import csv
import zipfile
import importlib.util
import sys
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

def load_custom_model_class(model_path, class_name):
    """Load custom model class from Python file"""
    print(f"Loading custom model class {class_name} from {model_path}")
    
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
# Define Default Segmentation Model (Fallback)
# ----------------------------
class DefaultSegmentationModel(nn.Module):
    def __init__(self, num_classes=21):
        super(DefaultSegmentationModel, self).__init__()
        
        # Encoder (VGG-like)
        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # Decoder (Upsampling path)
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            
            nn.Conv2d(64, num_classes, kernel_size=1)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return {'out': x}

# ----------------------------
# Load PyTorch Model
# ----------------------------
def load_model(weight_path, model_definition_path=None, model_class_name=None, num_classes=21):
    """Load custom segmentation model from .pth file"""
    print(f"Loading segmentation model from: {weight_path}")
    
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
        print("Trying to load state dict into default model as fallback")
        model = DefaultSegmentationModel(num_classes=num_classes)
        
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
        print("Loaded state dict into default fallback model")
        return model
    except Exception as e:
        print(f"Error loading fallback model: {e}")
    
    # Last resort: use a fresh default model
    print("Falling back to fresh default segmentation model")
    return DefaultSegmentationModel(num_classes=num_classes)

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
model = load_model(weight_path, model_definition_path, model_class_name, num_classes)
model.to(device)
model.eval()

# ----------------------------
# Set MLflow Tracking URI & Experiment
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
        if isinstance(output, dict) and 'out' in output:
            output = output['out']
        elif isinstance(output, torch.Tensor):
            # If model returns tensor directly
            pass
        else:
            print(f"Unexpected output format: {type(output)}")
            continue
        
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

