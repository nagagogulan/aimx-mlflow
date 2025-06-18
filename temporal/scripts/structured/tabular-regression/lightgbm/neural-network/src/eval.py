import os
import pandas as pd
import numpy as np
import json
import lightgbm as lgb
import mlflow
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_percentage_error, median_absolute_error

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

# Step 4: Load your trained LightGBM neural network model
# For neural network-like LightGBM, we use the gblinear booster
# which is saved in a text format
model = lgb.Booster(model_file=weight_path)

# Extract model configuration
model_config = {}
try:
    # Try to parse model configuration from the model file
    with open(weight_path, 'r') as f:
        for line in f:
            if line.startswith('parameters:'):
                config_str = line.replace('parameters:', '').strip()
                model_config = json.loads(config_str)
                break
except Exception as e:
    print(f"Warning: Could not parse model configuration: {e}")
    # Try to get configuration from the model object
    try:
        model_config = model.params
    except:
        print("Warning: Could not get model parameters")

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

# LightGBM neural network specific metrics
lgb_metrics = {}

# Get neural network specific parameters
lgb_metrics['model_type'] = 'neural_network'
lgb_metrics['booster'] = model_config.get('boosting', 'gblinear')
lgb_metrics['feature_selector'] = model_config.get('feature_selector', 'cyclic')
lgb_metrics['learning_rate'] = model_config.get('learning_rate', 0.1)
lgb_metrics['num_iterations'] = model_config.get('num_iterations', 100)
lgb_metrics['reg_lambda'] = model_config.get('lambda_l2', 0.0)
lgb_metrics['reg_alpha'] = model_config.get('lambda_l1', 0.0)

# Get coefficients (weights) for neural network-like model
try:
    # For gblinear booster, we can extract the linear weights
    if 'gblinear' in str(model_config.get('boosting', '')):
        # Get the number of features
        num_features = X.shape[1]
        
        # Get the linear coefficients
        linear_weights = model.dump_model().get('linear_weights', [])
        
        # Store weights in metrics
        if linear_weights and len(linear_weights) > 0:
            lgb_metrics['num_weights'] = len(linear_weights)
            
            # Calculate weight statistics
            weights_array = np.array(linear_weights)
            lgb_metrics['weights_mean'] = np.mean(weights_array)
            lgb_metrics['weights_std'] = np.std(weights_array)
            lgb_metrics['weights_min'] = np.min(weights_array)
            lgb_metrics['weights_max'] = np.max(weights_array)
            
            # Store top weights by magnitude
            weights_magnitude = np.abs(weights_array)
            top_indices = np.argsort(weights_magnitude)[-10:]  # Top 10
            
            for i, idx in enumerate(top_indices):
                if idx < len(X.columns):
                    feature_name = X.columns[idx]
                    lgb_metrics[f'top_weight_{i+1}_feature'] = feature_name
                    lgb_metrics[f'top_weight_{i+1}_value'] = float(linear_weights[idx])
except Exception as e:
    print(f"Warning: Could not extract neural network weights: {e}")

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

# Print LightGBM neural network specific metrics
print("\nLightGBM Neural Network Model Information:")
for metric_name, value in lgb_metrics.items():
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
    
    # Log LightGBM neural network specific metrics
    for metric_name, value in lgb_metrics.items():
        if isinstance(value, (int, float)) and not isinstance(value, dict):
            mlflow.log_metric(f"lgb_{metric_name}", value)
        elif isinstance(value, (str, bool)) and not isinstance(value, dict):
            mlflow.log_param(f"lgb_{metric_name}", value)
    
    # Log model parameters
    mlflow.log_param("model_path", weight_path)
    mlflow.log_param("dataset_path", dataset_path)
    
    # Log the LightGBM model
    try:
        mlflow.lightgbm.log_model(model, "model")
    except Exception as e:
        print(f"Warning: Could not log model to MLflow: {e}")
        # Try alternative method
        try:
            # Save model to a temporary file and log it as an artifact
            temp_model_path = "temp_model.txt"
            model.save_model(temp_model_path)
            mlflow.log_artifact(temp_model_path, "model")
            os.remove(temp_model_path)
        except Exception as e:
            print(f"Warning: Could not log model to MLflow using alternative method: {e}")
    
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
    
    # 5. Neural Network Weights Visualization (for gblinear booster)
    try:
        if 'gblinear' in str(model_config.get('boosting', '')):
            # Get the linear coefficients
            linear_weights = model.dump_model().get('linear_weights', [])
            
            if linear_weights and len(linear_weights) > 0:
                # Convert to numpy array
                weights_array = np.array(linear_weights)
                
                # Get feature names
                feature_names = list(X.columns)
                
                # Sort by absolute weight value
                sorted_indices = np.argsort(np.abs(weights_array))
                top_indices = sorted_indices[-20:]  # Top 20 weights
                
                # Get corresponding feature names and weights
                top_features = [feature_names[i] if i < len(feature_names) else f"Feature_{i}" for i in top_indices]
                top_weights = weights_array[top_indices]
                
                # Create a horizontal bar chart
                plt.figure(figsize=(12, 10))
                colors = ['blue' if w >= 0 else 'red' for w in top_weights]
                plt.barh(range(len(top_features)), top_weights, color=colors)
                plt.yticks(range(len(top_features)), top_features)
                plt.title('Top 20 Feature Weights (Neural Network Coefficients)')
                plt.xlabel('Weight Value')
                plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                plt.grid(axis='x', alpha=0.3)
                plt.tight_layout()
                
                # Save the plot
                weights_plot_path = "neural_network_weights.png"
                plt.savefig(weights_plot_path)
                plt.close()
                
                # Log to MLflow
                with mlflow.start_run(run_id=mlflow.active_run().info.run_id):
                    mlflow.log_artifact(weights_plot_path)
    except Exception as e:
        print(f"Warning: Could not create neural network weights visualization: {e}")
    
    # 6. Weight Distribution Histogram
    try:
        if 'gblinear' in str(model_config.get('boosting', '')):
            # Get the linear coefficients
            linear_weights = model.dump_model().get('linear_weights', [])
            
            if linear_weights and len(linear_weights) > 0:
                # Convert to numpy array
                weights_array = np.array(linear_weights)
                
                # Create a histogram of weights
                plt.figure(figsize=(10, 6))
                plt.hist(weights_array, bins=30, alpha=0.7, color='green')
                plt.title('Distribution of Neural Network Weights')
                plt.xlabel('Weight Value')
                plt.ylabel('Frequency')
                plt.axvline(x=0, color='red', linestyle='--')
                plt.grid(True, alpha=0.3)
                
                # Save the plot
                weight_dist_path = "weight_distribution.png"
                plt.savefig(weight_dist_path)
                plt.close()
                
                # Log to MLflow
                with mlflow.start_run(run_id=mlflow.active_run().info.run_id):
                    mlflow.log_artifact(weight_dist_path)
    except Exception as e:
        print(f"Warning: Could not create weight distribution visualization: {e}")
    
    # Log all plots to MLflow
    with mlflow.start_run(run_id=mlflow.active_run().info.run_id):
        mlflow.log_artifact(residuals_plot_path)
        mlflow.log_artifact(scatter_plot_path)
        mlflow.log_artifact(residuals_vs_predicted_path)
        mlflow.log_artifact(qq_plot_path)
    
    print("All visualizations created and logged to MLflow")
except Exception as e:
    print(f"Warning: Could not create visualizations: {e}")