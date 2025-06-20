import os
import pandas as pd
import numpy as np
import pickle
import mlflow
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_percentage_error, median_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
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

# Step 4: Load your trained scikit-learn neural network model
with open(weight_path, 'rb') as file:
    model = pickle.load(file)

# Step 5: Predict
predictions = model.predict(X)

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

# Neural network specific metrics
nn_metrics = {}

# Identify if the model is a pipeline or direct MLPRegressor
is_pipeline = isinstance(model, Pipeline)
if is_pipeline:
    nn_metrics['is_pipeline'] = True
    # Find the MLPRegressor in the pipeline
    for name, step in model.named_steps.items():
        if isinstance(step, MLPRegressor):
            mlp_model = step
            nn_metrics['mlp_step_name'] = name
            break
    else:
        # If no MLPRegressor found in pipeline
        mlp_model = None
        nn_metrics['mlp_in_pipeline'] = False
else:
    mlp_model = model if isinstance(model, MLPRegressor) else None

# Get model type
model_type = type(model).__name__
nn_metrics['model_type'] = model_type

# Extract neural network specific metrics if we have an MLPRegressor
if mlp_model is not None:
    # Architecture information
    nn_metrics['hidden_layer_sizes'] = str(mlp_model.hidden_layer_sizes)
    nn_metrics['n_layers'] = len(mlp_model.hidden_layer_sizes) + 1  # +1 for output layer
    nn_metrics['activation'] = mlp_model.activation
    nn_metrics['solver'] = mlp_model.solver
    nn_metrics['alpha'] = mlp_model.alpha
    nn_metrics['learning_rate'] = mlp_model.learning_rate
    nn_metrics['max_iter'] = mlp_model.max_iter
    
    # Training information if available
    if hasattr(mlp_model, 'n_iter_'):
        nn_metrics['n_iterations'] = mlp_model.n_iter_
    
    if hasattr(mlp_model, 'loss_'):
        nn_metrics['final_loss'] = mlp_model.loss_
    
    if hasattr(mlp_model, 'best_loss_'):
        nn_metrics['best_loss'] = mlp_model.best_loss_
    
    if hasattr(mlp_model, 'n_layers_'):
        nn_metrics['actual_n_layers'] = mlp_model.n_layers_
    
    if hasattr(mlp_model, 'n_outputs_'):
        nn_metrics['n_outputs'] = mlp_model.n_outputs_
    
    if hasattr(mlp_model, 'out_activation_'):
        nn_metrics['output_activation'] = mlp_model.out_activation_
    
    # Count total parameters
    if hasattr(mlp_model, 'coefs_') and hasattr(mlp_model, 'intercepts_'):
        total_params = sum(np.prod(coef.shape) for coef in mlp_model.coefs_)
        total_params += sum(np.prod(intercept.shape) for intercept in mlp_model.intercepts_)
        nn_metrics['total_parameters'] = total_params
        
        # Calculate weight statistics
        all_weights = np.concatenate([coef.flatten() for coef in mlp_model.coefs_])
        nn_metrics['weight_mean'] = np.mean(all_weights)
        nn_metrics['weight_std'] = np.std(all_weights)
        nn_metrics['weight_min'] = np.min(all_weights)
        nn_metrics['weight_max'] = np.max(all_weights)
        nn_metrics['weight_abs_mean'] = np.mean(np.abs(all_weights))
        
        # Calculate bias statistics
        all_biases = np.concatenate([intercept.flatten() for intercept in mlp_model.intercepts_])
        nn_metrics['bias_mean'] = np.mean(all_biases)
        nn_metrics['bias_std'] = np.std(all_biases)
        nn_metrics['bias_min'] = np.min(all_biases)
        nn_metrics['bias_max'] = np.max(all_biases)
        
        # Calculate layer-wise statistics
        for i, (coef, intercept) in enumerate(zip(mlp_model.coefs_, mlp_model.intercepts_)):
            nn_metrics[f'layer_{i+1}_shape'] = f"{coef.shape[0]}x{coef.shape[1]}"
            nn_metrics[f'layer_{i+1}_weight_mean'] = np.mean(coef)
            nn_metrics[f'layer_{i+1}_weight_std'] = np.std(coef)
            nn_metrics[f'layer_{i+1}_bias_mean'] = np.mean(intercept)
            nn_metrics[f'layer_{i+1}_bias_std'] = np.std(intercept)
            
            # Calculate weight magnitude distribution
            weight_magnitudes = np.abs(coef.flatten())
            nn_metrics[f'layer_{i+1}_weight_magnitude_mean'] = np.mean(weight_magnitudes)
            nn_metrics[f'layer_{i+1}_weight_magnitude_median'] = np.median(weight_magnitudes)
            nn_metrics[f'layer_{i+1}_weight_magnitude_max'] = np.max(weight_magnitudes)
            
            # Calculate sparsity (percentage of weights close to zero)
            sparsity = np.sum(np.abs(coef) < 0.01) / coef.size
            nn_metrics[f'layer_{i+1}_sparsity'] = sparsity

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
print(f"\nNeural Network Information ({model_type}):")
for metric_name, value in nn_metrics.items():
    if isinstance(value, (int, float, str, bool)):
        print(f"{metric_name}: {value}")

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
    mlflow.log_param("model_type", model_type)
    
    # Log the scikit-learn model
    try:
        mlflow.sklearn.log_model(model, "model")
    except Exception as e:
        print(f"Warning: Could not log model to MLflow: {e}")
    
    print(f"Metrics successfully logged to MLflow experiment: {experiment_name}")

# Create visualizations
try:
    import matplotlib.pyplot as plt
    
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
    
    # 5. Neural Network Architecture Visualization (if MLPRegressor)
    if mlp_model is not None and hasattr(mlp_model, 'coefs_'):
        # Create a visualization of the neural network architecture
        fig, ax = plt.figure(figsize=(12, 8)), plt.gca()
        
        # Get layer sizes
        layer_sizes = [mlp_model.coefs_[0].shape[0]]  # Input layer
        for coef in mlp_model.coefs_:
            layer_sizes.append(coef.shape[1])  # Hidden and output layers
        
        # Calculate positions
        n_layers = len(layer_sizes)
        v_spacing = 1
        h_spacing = 2
        
        # Determine the largest layer size for scaling
        largest_layer = max(layer_sizes)
        
        # Draw nodes
        for n, layer_size in enumerate(layer_sizes):
            layer_name = "Input" if n == 0 else "Output" if n == n_layers-1 else f"Hidden {n}"
            
            # Calculate vertical positions for nodes in this layer
            top_node_y = v_spacing * (layer_size - 1) / 2
            
            # Draw nodes
            for m in range(layer_size):
                node_x = n * h_spacing
                node_y = top_node_y - m * v_spacing
                
                # Draw the node
                circle = plt.Circle((node_x, node_y), 0.2, fill=True, color='lightblue')
                ax.add_patch(circle)
                
                # Add text label for layer type on first node of each layer
                if m == 0:
                    plt.text(node_x, top_node_y + 0.5, layer_name, ha='center')