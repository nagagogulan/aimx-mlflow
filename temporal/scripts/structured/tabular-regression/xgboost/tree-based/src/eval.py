import os
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
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

# Step 4: Load your trained XGBoost model
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

# XGBoost-specific metrics
xgb_metrics = {}

# Get feature importance
if hasattr(model, 'feature_importances_'):
    feature_importance = {feature: importance for feature, importance in 
                         zip(X.columns, model.feature_importances_)}
    
    # Sort feature importance
    feature_importance = dict(sorted(feature_importance.items(), 
                                    key=lambda item: item[1], 
                                    reverse=True))
else:
    feature_importance = {}

# Get model parameters
if hasattr(model, 'get_params'):
    model_params = model.get_params()
    xgb_metrics.update(model_params)

# Get number of trees/estimators
if hasattr(model, 'n_estimators'):
    xgb_metrics['n_estimators'] = model.n_estimators
elif hasattr(model, 'get_booster'):
    xgb_metrics['n_estimators'] = model.get_booster().num_boosted_rounds()

# Get tree depth if available
if hasattr(model, 'max_depth'):
    xgb_metrics['max_depth'] = model.max_depth

# Get learning rate if available
if hasattr(model, 'learning_rate'):
    xgb_metrics['learning_rate'] = model.learning_rate
elif hasattr(model, 'eta'):
    xgb_metrics['learning_rate'] = model.eta

# Get objective function if available
if hasattr(model, 'objective'):
    xgb_metrics['objective'] = model.objective

# Get booster type if available
if hasattr(model, 'booster'):
    xgb_metrics['booster'] = model.booster

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

# Print XGBoost-specific metrics
print("\nXGBoost Model Information:")
for metric_name, value in xgb_metrics.items():
    if isinstance(value, (int, float, str, bool)):
        print(f"{metric_name}: {value}")

# Print top feature importances
if feature_importance:
    print("\nTop Feature Importances:")
    for i, (feature, importance) in enumerate(feature_importance.items()):
        if i < 10:  # Show top 10 features
            print(f"{feature}: {importance:.4f}")

# Combine with original data for reference
output = data.copy()
output['Prediction'] = predictions
output['Absolute Error'] = abs(output[target_column] - predictions)
output['Squared Error'] = (output[target_column] - predictions) ** 2
output['Percentage Error'] = abs((output[target_column] - predictions) / output[target_column]) * 100

# Calculate residuals
residuals = y - predictions

# Print residual statistics
print("\nResidual Statistics:")
print(f"Mean of Residuals:      {np.mean(residuals):.4f}")
print(f"Std Dev of Residuals:   {np.std(residuals):.4f}")
print(f"Min Residual:           {np.min(residuals):.4f}")
print(f"Max Residual:           {np.max(residuals):.4f}")
print(f"25th Percentile:        {np.percentile(residuals, 25):.4f}")
print(f"Median of Residuals:    {np.median(residuals):.4f}")
print(f"75th Percentile:        {np.percentile(residuals, 75):.4f}")

# Log metrics to MLflow
# Set the MLflow tracking URI
mlflow.set_tracking_uri(mlflowURI)

# Set the experiment
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
    
    # Log feature importance
    for feature, importance in feature_importance.items():
        mlflow.log_metric(f"importance_{feature}", importance)
    
    # Log XGBoost-specific metrics
    for metric_name, value in xgb_metrics.items():
        if isinstance(value, (int, float, str, bool)):
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"xgb_{metric_name}", value)
            else:
                mlflow.log_param(f"xgb_{metric_name}", value)
    
    # Log model parameters
    mlflow.log_param("model_path", weight_path)
    mlflow.log_param("dataset_path", dataset_path)
    
    # Log the XGBoost model
    try:
        mlflow.xgboost.log_model(model, "model")
    except Exception as e:
        print(f"Warning: Could not log model to MLflow: {e}")
        # Try alternative method
        try:
            mlflow.sklearn.log_model(model, "model")
        except Exception as e:
            print(f"Warning: Could not log model to MLflow using sklearn: {e}")
    
    print(f"Metrics successfully logged to MLflow experiment: {experiment_name}")

# Create a histogram of residuals
try:
    import matplotlib.pyplot as plt
    
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
    
    # Log the plot to MLflow
    with mlflow.start_run(run_id=mlflow.active_run().info.run_id):
        mlflow.log_artifact(residuals_plot_path)
    
    print(f"Residuals histogram saved to {residuals_plot_path} and logged to MLflow")
except Exception as e:
    print(f"Warning: Could not create residuals histogram: {e}")

# Create a scatter plot of actual vs predicted values
try:
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
    
    # Log the plot to MLflow
    with mlflow.start_run(run_id=mlflow.active_run().info.run_id):
        mlflow.log_artifact(scatter_plot_path)
    
    print(f"Actual vs Predicted scatter plot saved to {scatter_plot_path} and logged to MLflow")
except Exception as e:
    print(f"Warning: Could not create scatter plot: {e}")

# Create a feature importance plot
try:
    if feature_importance:
        # Sort feature importances
        sorted_importances = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        features = [x[0] for x in sorted_importances[:10]]  # Top 10 features
        importances = [x[1] for x in sorted_importances[:10]]
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(features)), importances, align='center')
        plt.yticks(range(len(features)), features)
        plt.title('Top 10 Feature Importances')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        # Save the plot
        importance_plot_path = "feature_importance.png"
        plt.savefig(importance_plot_path)
        plt.close()
        
        # Log the plot to MLflow
        with mlflow.start_run(run_id=mlflow.active_run().info.run_id):
            mlflow.log_artifact(importance_plot_path)
        
        print(f"Feature importance plot saved to {importance_plot_path} and logged to MLflow")
except Exception as e:
    print(f"Warning: Could not create feature importance plot: {e}")