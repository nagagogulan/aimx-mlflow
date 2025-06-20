import os
import pandas as pd
import numpy as np
import mlflow
import tensorflow as tf
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_percentage_error, median_absolute_error
from sklearn.preprocessing import StandardScaler

# Load variables from .env file
load_dotenv()

mlflowURI = os.getenv("MLFLOW_TRACKING_URI")
weight_path = os.getenv("MODEL_WIGHTS_PATH")
dataset_path = os.getenv("DATASET_PATH")
target_column = os.getenv("TARGET_COLUMN")
experiment_name = os.getenv("EXPERIMENT_NAME")

# Step 1: Load the dataset
data = pd.read_csv(dataset_path)

# Step 2: Separate features and target
X = data.drop(columns=[target_column])
y = data[target_column]

# Step 3: Preprocess if needed
# Convert categorical features to numeric
for col in X.select_dtypes(include=['object']).columns:
    X[col] = pd.factorize(X[col])[0]

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Load your trained TensorFlow model
model = tf.keras.models.load_model(weight_path)

# Step 5: Predict
predictions = model.predict(X_scaled).flatten()

# Step 6: Calculate Evaluation Metrics
# Mean Absolute Error (MAE)
mae = mean_absolute_error(y, predictions)

# Mean Squared Error (MSE)
mse = mean_squared_error(y, predictions)

# Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)

# R-squared (Coefficient of Determination)
r2 = r2_score(y, predictions)

# Mean Absolute Percentage Error (MAPE)
mape = mean_absolute_percentage_error(y, predictions)

# Median Absolute Error (MedAE)
medae = median_absolute_error(y, predictions)

# Explained Variance Score
explained_variance = 1 - (np.var(y - predictions) / np.var(y))

# Mean Squared Log Error (MSLE) - only for positive values
msle = 0
try:
    if np.all(y > 0) and np.all(predictions > 0):
        msle = np.mean(np.square(np.log1p(y) - np.log1p(predictions)))
except Exception as e:
    print(f"Warning: Could not calculate MSLE: {e}")

# Max Error
max_error = np.max(np.abs(y - predictions))

# Calculate additional regression metrics
# Mean Bias Error (MBE)
mbe = np.mean(predictions - y)

# Relative Absolute Error (RAE)
rae = np.sum(np.abs(predictions - y)) / np.sum(np.abs(y - np.mean(y)))

# Relative Squared Error (RSE)
rse = np.sum(np.square(predictions - y)) / np.sum(np.square(y - np.mean(y)))

# Root Relative Squared Error (RRSE)
rrse = np.sqrt(rse)

# Normalized RMSE
nrmse_mean = rmse / np.mean(y)
nrmse_range = rmse / (np.max(y) - np.min(y))

# Calculate residuals
residuals = y - predictions

# Extract neural network specific metrics
nn_metrics = {}

# Get model architecture
nn_metrics['model_type'] = model.__class__.__name__
nn_metrics['num_layers'] = len(model.layers)

# Get layer information
layers_info = []
total_params = 0
trainable_params = 0
non_trainable_params = 0

for i, layer in enumerate(model.layers):
    layer_info = {
        'name': layer.name,
        'type': layer.__class__.__name__,
        'trainable': layer.trainable
    }
    
    # Get layer output shape
    if hasattr(layer, 'output_shape'):
        if isinstance(layer.output_shape, tuple):
            layer_info['output_shape'] = str(layer.output_shape)
        else:
            layer_info['output_shape'] = str(layer.output_shape[0])
    
    # Get layer parameters
    layer_params = layer.count_params()
    layer_info['params'] = layer_params
    total_params += layer_params
    
    if layer.trainable:
        trainable_params += layer_params
    else:
        non_trainable_params += layer_params
    
    # Get activation function
    if hasattr(layer, 'activation') and layer.activation is not None:
        if hasattr(layer.activation, '__name__'):
            layer_info['activation'] = layer.activation.__name__
        else:
            layer_info['activation'] = str(layer.activation)
    
    # Add to layers list
    layers_info.append(layer_info)
    
    # Add summary to metrics
    nn_metrics[f'layer_{i+1}_type'] = layer_info['type']
    nn_metrics[f'layer_{i+1}_name'] = layer_info['name']
    if 'output_shape' in layer_info:
        nn_metrics[f'layer_{i+1}_shape'] = layer_info['output_shape']
    nn_metrics[f'layer_{i+1}_params'] = layer_info['params']
    if 'activation' in layer_info:
        nn_metrics[f'layer_{i+1}_activation'] = layer_info['activation']

# Add total parameters to metrics
nn_metrics['total_params'] = total_params
nn_metrics['trainable_params'] = trainable_params
nn_metrics['non_trainable_params'] = non_trainable_params

# Get optimizer information
if hasattr(model, 'optimizer') and model.optimizer is not None:
    optimizer = model.optimizer
    nn_metrics['optimizer_type'] = optimizer.__class__.__name__
    
    # Get optimizer parameters
    if hasattr(optimizer, 'get_config'):
        optimizer_config = optimizer.get_config()
        for key, value in optimizer_config.items():
            if isinstance(value, (int, float, str, bool)):
                nn_metrics[f'optimizer_{key}'] = value

# Get loss function
if hasattr(model, 'loss') and model.loss is not None:
    if isinstance(model.loss, dict):
        nn_metrics['loss_function'] = str(list(model.loss.values())[0])
    else:
        nn_metrics['loss_function'] = str(model.loss)

# Get metrics used during training
if hasattr(model, 'metrics') and model.metrics is not None:
    metrics_list = []
    for metric in model.metrics:
        if hasattr(metric, 'name'):
            metrics_list.append(metric.name)
        else:
            metrics_list.append(str(metric))
    nn_metrics['training_metrics'] = ', '.join(metrics_list)

# Print evaluation metrics
print("\nEvaluation Metrics:")
print(f"MAE (Mean Absolute Error):       {mae:.4f}")
print(f"MSE (Mean Squared Error):        {mse:.4f}")
print(f"RMSE (Root Mean Squared Error):  {rmse:.4f}")
print(f"R-Squared (R²):                  {r2:.4f}")
print(f"MAPE (Mean Absolute Percentage Error): {mape:.4f}")
print(f"MedAE (Median Absolute Error):   {medae:.4f}")
print(f"Explained Variance Score:        {explained_variance:.4f}")
if msle > 0:
    print(f"MSLE (Mean Squared Log Error):  {msle:.4f}")
print(f"Max Error:                       {max_error:.4f}")
print(f"MBE (Mean Bias Error):           {mbe:.4f}")
print(f"RAE (Relative Absolute Error):   {rae:.4f}")
print(f"RSE (Relative Squared Error):    {rse:.4f}")
print(f"RRSE (Root Relative Squared Error): {rrse:.4f}")
print(f"NRMSE (Mean):                    {nrmse_mean:.4f}")
print(f"NRMSE (Range):                   {nrmse_range:.4f}")

# Print neural network specific metrics
print("\nNeural Network Information:")
print(f"Model Type: {nn_metrics['model_type']}")
print(f"Number of Layers: {nn_metrics['num_layers']}")
print(f"Total Parameters: {nn_metrics['total_params']}")
print(f"Trainable Parameters: {nn_metrics['trainable_params']}")
print(f"Non-trainable Parameters: {nn_metrics['non_trainable_params']}")

if 'optimizer_type' in nn_metrics:
    print(f"Optimizer: {nn_metrics['optimizer_type']}")

if 'loss_function' in nn_metrics:
    print(f"Loss Function: {nn_metrics['loss_function']}")

if 'training_metrics' in nn_metrics:
    print(f"Training Metrics: {nn_metrics['training_metrics']}")

print("\nLayer Information:")
for i in range(nn_metrics['num_layers']):
    layer_idx = i + 1
    layer_type = nn_metrics.get(f'layer_{layer_idx}_type', 'Unknown')
    layer_name = nn_metrics.get(f'layer_{layer_idx}_name', 'Unknown')
    layer_shape = nn_metrics.get(f'layer_{layer_idx}_shape', 'Unknown')
    layer_params = nn_metrics.get(f'layer_{layer_idx}_params', 0)
    layer_activation = nn_metrics.get(f'layer_{layer_idx}_activation', 'None')
    
    print(f"Layer {layer_idx}: {layer_type} ('{layer_name}') - Shape: {layer_shape}, Parameters: {layer_params}, Activation: {layer_activation}")

# Log metrics to MLflow
mlflow.set_tracking_uri(mlflowURI)
mlflow.set_experiment(experiment_name)

# Start an MLflow run
with mlflow.start_run():
    # Log the metrics
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mape", mape)
    mlflow.log_metric("medae", medae)
    mlflow.log_metric("explained_variance", explained_variance)
    if msle > 0:
        mlflow.log_metric("msle", msle)
    mlflow.log_metric("max_error", max_error)
    mlflow.log_metric("mbe", mbe)
    mlflow.log_metric("rae", rae)
    mlflow.log_metric("rse", rse)
    mlflow.log_metric("rrse", rrse)
    mlflow.log_metric("nrmse_mean", nrmse_mean)
    mlflow.log_metric("nrmse_range", nrmse_range)
    
    # Log residual statistics
    mlflow.log_metric("residuals_mean", np.mean(residuals))
    mlflow.log_metric("residuals_std", np.std(residuals))
    mlflow.log_metric("residuals_min", np.min(residuals))
    mlflow.log_metric("residuals_max", np.max(residuals))
    mlflow.log_metric("residuals_25th", np.percentile(residuals, 25))
    mlflow.log_metric("residuals_median", np.median(residuals))
    mlflow.log_metric("residuals_75th", np.percentile(residuals, 75))
    
    # Log prediction statistics
    mlflow.log_metric("predictions_mean", np.mean(predictions))
    mlflow.log_metric("predictions_std", np.std(predictions))
    mlflow.log_metric("predictions_min", np.min(predictions))
    mlflow.log_metric("predictions_max", np.max(predictions))
    
    # Log target statistics
    mlflow.log_metric("target_mean", np.mean(y))
    mlflow.log_metric("target_std", np.std(y))
    mlflow.log_metric("target_min", np.min(y))
    mlflow.log_metric("target_max", np.max(y))
    
    # Log neural network specific metrics
    for metric_name, value in nn_metrics.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            mlflow.log_metric(f"nn_{metric_name}", value)
        elif isinstance(value, (str, bool)):
            mlflow.log_param(f"nn_{metric_name}", value)
    
    # Log model parameters
    mlflow.log_param("model_path", weight_path)
    mlflow.log_param("dataset_path", dataset_path)
    
    # Log the TensorFlow model
    try:
        # Save model signature
        signature = mlflow.models.signature.infer_signature(X_scaled, predictions)
        
        # Log the model
        mlflow.tensorflow.log_model(
            model, 
            "model",
            signature=signature
        )
    except Exception as e:
        print(f"Warning: Could not log model to MLflow: {e}")
    
    print(f"Metrics successfully logged to MLflow experiment: {experiment_name}")

# Create visualizations
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # 1. Histogram of residuals
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=30, alpha=0.7, color='blue')
    plt.title('Histogram of Residuals')
    plt.xlabel('Residual Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    residuals_plot_path = "residuals_histogram.png"
    plt.savefig(residuals_plot_path)
    plt.close()
    
    # 2. Scatter plot of actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y, predictions, alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.title('Actual vs Predicted Values')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    scatter_plot_path = "actual_vs_predicted.png"
    plt.savefig(scatter_plot_path)
    plt.close()
    
    # 3. Residuals vs Predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(predictions, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Residuals vs Predicted Values')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    residuals_vs_predicted_path = "residuals_vs_predicted.png"
    plt.savefig(residuals_vs_predicted_path)
    plt.close()
    
    # 4. QQ Plot for residuals normality check
    from scipy import stats
    
    plt.figure(figsize=(10, 6))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot of Residuals', fontsize=15)
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    qq_plot_path = "residuals_qq_plot.png"
    plt.savefig(qq_plot_path)
    plt.close()
    
    # 5. Neural Network Architecture Visualization
    # Create a simplified visualization of the neural network architecture
    try:
        # Get layer sizes
        layer_sizes = []
        layer_types = []
        
        for layer in model.layers:
            if hasattr(layer, 'output_shape'):
                if isinstance(layer.output_shape, tuple):
                    output_shape = layer.output_shape
                else:
                    output_shape = layer.output_shape[0]
                
                # Get the last dimension as the layer size
                if len(output_shape) > 1:
                    layer_sizes.append(output_shape[-1])
                else:
                    layer_sizes.append(output_shape[0])
            else:
                layer_sizes.append(0)
            
            layer_types.append(layer.__class__.__name__)
        
        # Create a figure for the neural network architecture
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Constants for drawing
        n_layers = len(layer_sizes)
        v_spacing = 0.25
        h_spacing = 1
        
        # Determine the largest layer size for scaling
        largest_layer = max(layer_sizes) if layer_sizes else 1
        v_margin = 0.25
        
        # Calculate total height and width
        layer_heights = [size / largest_layer * (1 - 2 * v_margin) + v_margin for size in layer_sizes]
        total_height = max(layer_heights) + v_spacing
        total_width = h_spacing * (n_layers - 1) + 2  # Add margin on both sides
        
        # Set axis limits
        ax.set_xlim(0, total_width)
        ax.set_ylim(0, total_height)
        
        # Remove axis ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        
        # Draw layers
        layer_positions = []
        for i, (size, height, layer_type