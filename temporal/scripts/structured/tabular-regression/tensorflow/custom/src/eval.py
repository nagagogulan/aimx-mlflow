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
# For custom models, we need to handle different possible formats
try:
    # Try loading as a SavedModel
    model = tf.saved_model.load(weight_path)
    is_saved_model = True
    print("Loaded model as a SavedModel")
except Exception as e:
    print(f"Could not load as SavedModel: {e}")
    try:
        # Try loading as a Keras model
        model = tf.keras.models.load_model(weight_path)
        is_saved_model = False
        print("Loaded model as a Keras model")
    except Exception as e:
        print(f"Could not load as Keras model: {e}")
        # Try loading as a custom format
        try:
            # Check if there's a custom loading script
            import importlib.util
            spec = importlib.util.spec_from_file_location("model_loader", os.path.join(weight_path, "loader.py"))
            if spec:
                model_loader = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(model_loader)
                model = model_loader.load_model(weight_path)
                is_saved_model = False
                print("Loaded model using custom loader")
            else:
                raise Exception("No custom loader found")
        except Exception as e:
            print(f"Could not load using custom loader: {e}")
            raise Exception(f"Failed to load model from {weight_path}: {e}")

# Step 5: Predict
if is_saved_model:
    # For SavedModel format
    try:
        # Try using the serving signature
        infer = model.signatures["serving_default"]
        predictions = []
        # Process in batches to avoid memory issues
        batch_size = 32
        for i in range(0, len(X_scaled), batch_size):
            batch = X_scaled[i:i+batch_size]
            tensor = tf.convert_to_tensor(batch, dtype=tf.float32)
            batch_preds = infer(tensor)
            # Extract the output tensor (assuming it's the first or only output)
            output_key = list(batch_preds.keys())[0]
            batch_preds = batch_preds[output_key].numpy()
            predictions.extend(batch_preds.flatten())
        predictions = np.array(predictions)
    except Exception as e:
        print(f"Error using serving signature: {e}")
        # Try calling the model directly
        try:
            predictions = []
            batch_size = 32
            for i in range(0, len(X_scaled), batch_size):
                batch = X_scaled[i:i+batch_size]
                tensor = tf.convert_to_tensor(batch, dtype=tf.float32)
                batch_preds = model(tensor).numpy()
                predictions.extend(batch_preds.flatten())
            predictions = np.array(predictions)
        except Exception as e:
            print(f"Error calling model directly: {e}")
            raise Exception(f"Failed to generate predictions: {e}")
else:
    # For Keras model
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

# Extract model specific metrics
model_metrics = {}

# Try to get model information
if not is_saved_model and hasattr(model, 'summary'):
    # For Keras models
    model_metrics['model_type'] = model.__class__.__name__
    model_metrics['num_layers'] = len(model.layers)
    
    # Get layer information
    total_params = 0
    trainable_params = 0
    non_trainable_params = 0
    
    for i, layer in enumerate(model.layers):
        layer_info = {
            'name': layer.name,
            'type': layer.__class__.__name__,
            'trainable': layer.trainable
        }
        
        # Get layer parameters
        layer_params = layer.count_params()
        total_params += layer_params
        
        if layer.trainable:
            trainable_params += layer_params
        else:
            non_trainable_params += layer_params
        
        # Add to metrics
        model_metrics[f'layer_{i+1}_type'] = layer_info['type']
        model_metrics[f'layer_{i+1}_name'] = layer_info['name']
        model_metrics[f'layer_{i+1}_params'] = layer_params
    
    # Add total parameters to metrics
    model_metrics['total_params'] = total_params
    model_metrics['trainable_params'] = trainable_params
    model_metrics['non_trainable_params'] = non_trainable_params
    
    # Get optimizer information
    if hasattr(model, 'optimizer') and model.optimizer is not None:
        optimizer = model.optimizer
        model_metrics['optimizer_type'] = optimizer.__class__.__name__
        
        # Get optimizer parameters
        if hasattr(optimizer, 'get_config'):
            optimizer_config = optimizer.get_config()
            for key, value in optimizer_config.items():
                if isinstance(value, (int, float, str, bool)):
                    model_metrics[f'optimizer_{key}'] = value
    
    # Get loss function
    if hasattr(model, 'loss') and model.loss is not None:
        if isinstance(model.loss, dict):
            model_metrics['loss_function'] = str(list(model.loss.values())[0])
        else:
            model_metrics['loss_function'] = str(model.loss)
else:
    # For SavedModel or custom models
    model_metrics['model_type'] = 'SavedModel' if is_saved_model else 'Custom'
    
    # Try to get signature information
    if is_saved_model:
        try:
            signatures = list(model.signatures.keys())
            model_metrics['signatures'] = ', '.join(signatures)
            
            # Try to get input and output specs
            signature = model.signatures['serving_default']
            input_specs = []
            for input_name, input_tensor in signature.inputs.items():
                input_specs.append(f"{input_name}: {input_tensor.shape} ({input_tensor.dtype})")
            model_metrics['input_specs'] = '; '.join(input_specs)
            
            output_specs = []
            for output_name, output_tensor in signature.outputs.items():
                output_specs.append(f"{output_name}: {output_tensor.shape} ({output_tensor.dtype})")
            model_metrics['output_specs'] = '; '.join(output_specs)
        except Exception as e:
            print(f"Warning: Could not extract signature information: {e}")

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

# Print model specific metrics
print("\nModel Information:")
for key, value in model_metrics.items():
    print(f"{key}: {value}")

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
    
    # Log model specific metrics
    for metric_name, value in model_metrics.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            mlflow.log_metric(f"model_{metric_name}", value)
        elif isinstance(value, (str, bool)):
            mlflow.log_param(f"model_{metric_name}", value)
    
    # Log model parameters
    mlflow.log_param("model_path", weight_path)
    mlflow.log_param("dataset_path", dataset_path)
    
    # Log the TensorFlow model
    try:
        if not is_saved_model:
            # Save model signature
            signature = mlflow.models.signature.infer_signature(X_scaled, predictions)
            
            # Log the model
            mlflow.tensorflow.log_model(
                model, 
                "model",
                signature=signature
            )
        else:
            # For SavedModel, we can log the model directory
            mlflow.log_artifact(weight_path, "saved_model")
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
    
    # 5. Prediction Error Distribution
    plt.figure(figsize=(10, 6))
    error_pct = (predictions - y) / y * 100
    plt.hist(error_pct, bins=30, alpha=0.7, color='green')
    plt.title('Prediction Error Distribution (% Error)')
    plt.xlabel('Percentage Error')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    error_dist_path = "error_distribution.png"
    plt.savefig(error_dist_path)
    plt.close()
    
    # 6. Feature Importance (if available)
    # For custom models, feature importance might not be directly available
    # We can use permutation importance as an alternative
    try:
        from sklearn.inspection import permutation_importance
        
        # Calculate permutation importance
        result = permutation_importance(
            lambda X: model.predict(X) if not is_saved_model else np.array([
                model(tf.convert_to_tensor(x.reshape(1, -1), dtype=tf.float32)).numpy().flatten()[0]
                for x in X
            ]),
            X_scaled,
            y,
            n_repeats=5,
            random_state=42,
            n_jobs=-1
        )
        
        # Get feature importances
        importances = result.importances_mean
        
        # Create a DataFrame for visualization
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importances
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # Plot feature importances
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title('Feature Importance (Permutation)')
        plt.tight_layout()
        
        # Save the plot
        importance_plot_path = "feature_importance.png"
        plt.savefig(importance_plot_path)
        plt.close()
        
        # Log to MLflow
        with mlflow.start_run(run_id=mlflow.active_run().info.run_id):
            mlflow.log_artifact(importance_plot_path)
            
            # Log feature importances as parameters
            for i, row in importance_df.head(10).iterrows():
                mlflow.log_metric(f"feature_importance_{row['Feature']}", row['Importance'])
    except Exception as e:
        print(f"Warning: Could not calculate feature importance: {e}")
    
    # 7. Correlation Heatmap
    plt.figure(figsize=(12, 10))
    
    # Add predictions to the dataset for correlation analysis
    data_with_preds = X.copy()
    data_with_preds[target_column] = y
    data_with_preds['Predictions'] = predictions
    
    # Calculate correlation matrix
    corr_matrix = data_with_preds.corr()
    
    # Plot heatmap
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    
    # Save the plot
    corr_plot_path = "correlation_heatmap.png"
    plt.savefig(corr_plot_path)
    plt.close()
    
    # Log all plots to MLflow
    with mlflow.start_run(run_id=mlflow.active_run().info.run_id):
        mlflow.log_artifact(residuals_plot_path)
        mlflow.log_artifact(scatter_plot_path)
        mlflow.log_artifact(residuals_vs_predicted_path)
        mlflow.log_artifact(qq_plot_path)
        mlflow.log_artifact(error_dist_path)
        mlflow.log_artifact(corr_plot_path)
    
except Exception as e:
    print(f"Warning: Could not create visualizations: {e}")

print("Evaluation completed successfully!")