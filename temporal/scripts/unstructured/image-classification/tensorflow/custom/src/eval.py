import os
import csv
import zipfile
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import mlflow
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import shutil
import matplotlib.pyplot as plt
import json

# Load variables from .env file
load_dotenv()

mlflowURI = os.getenv("MLFLOW_TRACKING_URI")
weight_path = os.getenv("MODEL_WEIGHTS_PATH")
images_zip_path = os.getenv("IMAGES_ZIP_PATH")
dataset_path = os.getenv("DATASET_PATH")
labels_path = os.getenv("DATA_LABELS_PATH")
experiment_name = os.getenv("EXPERIMENT_NAME")
input_size = int(os.getenv("INPUT_SIZE", "224"))
model_config_path = os.getenv("MODEL_CONFIG_PATH", "./model_config.json")
num_classes = int(os.getenv("NUM_CLASSES", "1000"))

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
def preprocess_image(image_path, input_size=(224, 224), rescale=True):
    """Preprocess image for model inference"""
    img = load_img(image_path, target_size=input_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Apply rescaling if needed
    if rescale:
        img_array = img_array / 255.0
    
    return img_array

def load_labels_csv(csv_path):
    """Load image labels from CSV file"""
    label_map = {}
    with open(csv_path, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = os.path.basename(row['image_path'])
            label_map[path] = int(row['label'])
    return label_map

def load_class_names(labels_path):
    """Load class names from file"""
    with open(labels_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names

def load_model_config(config_path):
    """Load model configuration from JSON file"""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}

# ----------------------------
# Load Model Configuration
# ----------------------------
model_config = load_model_config(model_config_path)
preprocessing_config = model_config.get('preprocessing', {})
rescale = preprocessing_config.get('rescale', True)
custom_objects = model_config.get('custom_objects', {})

# Convert string custom objects to actual classes/functions
for key, value in custom_objects.items():
    if value == "tf.keras.layers.LeakyReLU":
        custom_objects[key] = tf.keras.layers.LeakyReLU
    # Add more mappings as needed

# ----------------------------
# Load TensorFlow Model
# ----------------------------
print(f"Loading custom model from: {weight_path}")
try:
    # Load the model with custom objects if specified
    model = load_model(weight_path, custom_objects=custom_objects if custom_objects else None)
    model.summary()
    
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
    print("Attempting to create a basic CNN model...")
    try:
        # Create a basic CNN model as fallback
        from tensorflow.keras.applications import MobileNetV2
        
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(input_size, input_size, 3))
        x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        x = tf.keras.layers.Dense(1024, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
        
        print("Created fallback MobileNetV2 model")
        model.summary()
    except Exception as e2:
        print(f"Failed to create fallback model: {e2}")
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
all_preds = []
all_labels = []
all_probs = []
class_correct = {}
class_total = {}

print("Running inference...")
for image_file in image_files:
    # Preprocess image
    input_data = preprocess_image(image_file, input_size=(input_size, input_size), rescale=rescale)
    
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
        
        # Track per-class accuracy
        class_total[true_label] = class_total.get(true_label, 0) + 1
        
        if top_class == true_label:
            correct += 1
            class_correct[true_label] = class_correct.get(true_label, 0) + 1
        
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

# Calculate precision, recall, F1 score (macro average)
if len(all_preds) > 0:
    macro_precision = precision_score(all_labels, all_preds, average='macro')
    macro_recall = recall_score(all_labels, all_preds, average='macro')
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    print(f"Precision (macro): {macro_precision:.4f}")
    print(f"Recall (macro): {macro_recall:.4f}")
    print(f"F1 Score (macro): {macro_f1:.4f}")
    
    # Calculate per-class metrics
    classes = np.unique(all_labels)
    print("\nPer-class metrics:")
    for cls in classes:
        cls_precision = precision_score(all_labels, all_preds, labels=[cls], average='micro')
        cls_recall = recall_score(all_labels, all_preds, labels=[cls], average='micro')
        cls_f1 = f1_score(all_labels, all_preds, labels=[cls], average='micro')
        cls_accuracy = class_correct.get(cls, 0) / class_total.get(cls, 1)
        
        cls_name = class_names[cls] if cls < len(class_names) else f"Class {cls}"
        print(f"{cls_name}: Accuracy={cls_accuracy:.4f}, Precision={cls_precision:.4f}, Recall={cls_recall:.4f}, F1={cls_f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')

# ----------------------------
# Feature Visualization
# ----------------------------
try:
    # Select a sample image for visualization
    if len(image_files) > 0:
        sample_image = image_files[0]
        sample_input = preprocess_image(sample_image, input_size=(input_size, input_size), rescale=rescale)
        
        # Create a model that outputs the activations of the last convolutional layer
        conv_layers = [layer for layer in model.layers if 'conv' in layer.name.lower()]
        if conv_layers:
            last_conv_layer = conv_layers[-1]
            feature_model = tf.keras.Model(
                inputs=model.inputs,
                outputs=last_conv_layer.output
            )
            
            # Get feature maps
            feature_maps = feature_model.predict(sample_input)
            
            # Plot feature maps
            os.makedirs("./feature_maps", exist_ok=True)
            
            # Display the original image
            plt.figure(figsize=(10, 10))
            plt.imshow(load_img(sample_image, target_size=(input_size, input_size)))
            plt.title("Original Image")
            plt.axis('off')
            plt.savefig("./feature_maps/original_image.png")
            
            # Display feature maps (first 16 or fewer)
            num_features = min(16, feature_maps.shape[-1])
            plt.figure(figsize=(20, 10))
            for i in range(num_features):
                plt.subplot(4, 4, i+1)
                plt.imshow(feature_maps[0, :, :, i], cmap='viridis')
                plt.axis('off')
            plt.tight_layout()
            plt.savefig("./feature_maps/feature_maps.png")
            
            print("✅ Saved feature map visualization")
except Exception as e:
    print(f"Could not generate feature visualization: {e}")

# ----------------------------
# Class Activation Mapping (CAM)
# ----------------------------
try:
    # Implement Grad-CAM for visualization
    if len(image_files) > 0:
        from tensorflow.keras.models import Model
        
        # Find the last convolutional layer
        last_conv_layer = None
        for layer in reversed(model.layers):
            if 'conv' in layer.name.lower():
                last_conv_layer = layer
                break
        
        if last_conv_layer:
            # Create a model that maps the input image to the activations
            # of the last conv layer and the output predictions
            grad_model = Model(
                inputs=[model.inputs],
                outputs=[last_conv_layer.output, model.output]
            )
            
            # Choose a sample image
            sample_image = image_files[0]
            sample_input = preprocess_image(sample_image, input_size=(input_size, input_size), rescale=rescale)
            
            # Compute the gradient of the top predicted class for our sample image
            # with respect to the activations of the last conv layer
            with tf.GradientTape() as tape:
                conv_output, predictions = grad_model(sample_input)
                top_pred_idx = tf.argmax(predictions[0])
                top_class_score = predictions[0, top_pred_idx]
            
            # Gradient of the top predicted class with respect to the output feature map
            grads = tape.gradient(top_class_score, conv_output)
            
            # Vector of mean intensity of the gradient over a specific feature map channel
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            # Weight the channels by corresponding gradients
            conv_output = conv_output[0]
            heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_output), axis=-1)
            
            # For visualization purpose, normalize the heatmap between 0 & 1
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            heatmap = heatmap.numpy()
            
            # Resize the heatmap to the original image size
            import cv2
            img = cv2.imread(sample_image)
            img = cv2.resize(img, (input_size, input_size))
            heatmap = cv2.resize(heatmap, (input_size, input_size))
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            # Superimpose the heatmap on original image
            superimposed_img = heatmap * 0.4 + img
            superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')
            
            # Save the superimposed image
            os.makedirs("./cam_visualization", exist_ok=True)
            cv2.imwrite("./cam_visualization/cam.jpg", superimposed_img)
            
            print("✅ Saved Class Activation Map visualization")
except Exception as e:
    print(f"Could not generate CAM visualization: {e}")

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
        if 'macro_precision' in locals():
            mlflow.log_metric("macro_precision", macro_precision)
            mlflow.log_metric("macro_recall", macro_recall)
            mlflow.log_metric("macro_f1", macro_f1)
        
        # Log per-class metrics
        for cls in classes:
            cls_accuracy = class_correct.get(cls, 0) / class_total.get(cls, 1)
            mlflow.log_metric(f"class_{cls}_accuracy", cls_accuracy)
        
        # Log parameters
        mlflow.log_param("model_path", weight_path)
        mlflow.log_param("total_samples", total)
        mlflow.log_param("num_classes", len(np.unique(all_labels)))
        mlflow.log_param("input_size", input_size)
        
        # Log visualizations
        if os.path.exists('confusion_matrix.png'):
            mlflow.log_artifact('confusion_matrix.png', "visualizations")
        
        if os.path.exists('./feature_maps'):
            mlflow.log_artifacts("./feature_maps", "visualizations/feature_maps")
            
        if os.path.exists('./cam_visualization'):
            mlflow.log_artifacts("./cam_visualization", "visualizations/cam")
        
        print("\n✅ All metrics have been logged to MLflow")
else:
    print("\n⚠️ MLflow tracking URI not set, metrics not logged")

# ----------------------------
# Cleanup
# ----------------------------
shutil.rmtree(image_dir)
if os.path.exists("./feature_maps"):
    shutil.rmtree("./feature_maps")
if os.path.exists("./cam_visualization"):
    shutil.rmtree("./cam_visualization")
print("Temporary files cleaned up")