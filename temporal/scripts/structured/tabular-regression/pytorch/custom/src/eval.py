import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import mlflow
from dotenv import load_dotenv
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Load variables from .env file
load_dotenv()

mlflowURI = os.getenv("MLFLOW_TRACKING_URI")
weight_path = os.getenv("MODEL_WIGHTS_PATH")
dataset_path = os.getenv("DATASET_PATH")
target_column = os.getenv("TARGET_COLUMN")
experiment_name = os.getenv("EXPERIMENT_NAME")

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 1: Load the dataset
df = pd.read_csv(dataset_path)

# Step 2: Separate features and target
X = df.drop(columns=[target_column])
y = df[target_column]

# Step 3: Preprocess data for PyTorch
# Convert categorical features to numeric
for col in X.select_dtypes(include=['object']).columns:
    X[col] = pd.factorize(X[col])[0]

# Normalize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Create PyTorch dataset
class TabularDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets.values, dtype=torch.float32).reshape(-1, 1)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# Create dataset and dataloader
dataset = TabularDataset(X_scaled, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# Step 4: Load your trained model
# The model is assumed to be a custom PyTorch model saved with torch.save()
model = torch.load(weight_path, map_location=device)
model.eval()  # Set to evaluation mode

# Step 5: Predict
predictions = []
actuals = []

with torch.no_grad():
    for features, targets in dataloader:
        features = features.to(device)
        outputs = model(features)
        
        predictions.append(outputs.cpu().numpy())
        actuals.append(targets.numpy())

# Concatenate batch results
predictions = np.concatenate(predictions).flatten()
actuals = np.concatenate(actuals).flatten()

# Step 6: Calculate Evaluation Metrics
# Mean Squared Error (MSE)
mse = mean_squared_error(actuals, predictions)

# Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)

# Mean Absolute Error (MAE)
mae = mean_absolute_error(actuals, predictions)

# R-squared (Coefficient of Determination)
r2 = r2_score(actuals, predictions)

# Mean Absolute Percentage Error (MAPE)
# Handle division by zero
mape = np.mean(np.abs((actuals - predictions) / np.where(actuals == 0, 1e-10, actuals))) * 100

# Explained Variance Score
explained_variance = 1 - (np.var(actuals - predictions) / np.var(actuals))

# Calculate residuals
residuals = actuals - predictions

# Calculate additional regression metrics
# Mean Bias Error (MBE)
mbe = np.mean(predictions - actuals)

# Relative Absolute Error (RAE)
rae = np.sum(np.abs(predictions - actuals)) / np.sum(np.abs(actuals - np.mean(actuals)))

# Relative Squared Error (RSE)
rse = np.sum(np.square(predictions - actuals)) / np.sum(np.square(actuals - np.mean(actuals)))

# Root Relative Squared Error (RRSE)
rrse = np.sqrt(rse)

# Normalized RMSE
nrmse_mean = rmse / np.mean(actuals)
nrmse_range = rmse / (np.max(actuals) - np.min(actuals))

# Median Absolute Error
median_ae = np.median(np.abs(actuals - predictions))

# Print evaluation metrics
print("\nEvaluation Metrics:")
print(f"MSE (Mean Squared Error): {mse:.4f}")
print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
print(f"MAE (Mean Absolute Error): {mae:.4f}")
print(f"R² (Coefficient of Determination): {r2:.4f}")
print(f"MAPE (Mean Absolute Percentage Error): {mape:.4f}%")
print(f"Explained Variance: {explained_variance:.4f}")
print(f"MBE (Mean Bias Error): {mbe:.4f}")
print(f"RAE (Relative Absolute Error): {rae:.4f}")
print(f"RSE (Relative Squared Error): {rse:.4f}")
print(f"RRSE (Root Relative Squared Error): {rrse:.4f}")
print(f"NRMSE (Mean): {nrmse_mean:.4f}")
print(f"NRMSE (Range): {nrmse_range:.4f}")
print(f"Median Absolute Error: {median_ae:.4f}")

# Step 7: Create results dataframe
results = pd.DataFrame({
    "Actual": actuals,
    "Predicted": predictions,
    "Absolute Error": np.abs(actuals - predictions),
    "Squared Error": np.square(actuals - predictions),
    "Percentage Error": np.abs((actuals - predictions) / np.where(actuals == 0, 1e-10, actuals)) * 100
})

print("\nResults Summary:")
print(results.describe())

# Step 8: Log metrics to MLflow
# Set the MLflow tracking URI
mlflow.set_tracking_uri(mlflowURI)

# Set the experiment
mlflow.set_experiment(experiment_name)

# Start an MLflow run
with mlflow.start_run():
    # Log basic metrics
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mape", mape)
    mlflow.log_metric("explained_variance", explained_variance)
    mlflow.log_metric("median_absolute_error", median_ae)
    
    # Log additional metrics
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
    mlflow.log_metric("target_mean", np.mean(actuals))
    mlflow.log_metric("target_std", np.std(actuals))
    mlflow.log_metric("target_min", np.min(actuals))
    mlflow.log_metric("target_max", np.max(actuals))
    
    # Log model parameters
    mlflow.log_param("model_path", weight_path)
    mlflow.log_param("dataset_path", dataset_path)
    mlflow.log_param("device", str(device))
    
    # Try to extract model architecture information
    try:
        model_info = {
            "model_type": model.__class__.__name__,
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "num_trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
        
        # Log model info
        for key, value in model_info.items():
            mlflow.log_param(key, value)
    except Exception as e:
        print(f"Warning: Could not extract model architecture information: {e}")
    
    # Log the PyTorch model
    try:
        mlflow.pytorch.log_model(model, "model")
    except Exception as e:
        print(f"Warning: Could not log PyTorch model to MLflow: {e}")
    
    # Create visualizations
    try:
        # 1. Actual vs Predicted scatter plot
        plt.figure(figsize=(10, 6))
        plt.scatter(actuals, predictions, alpha=0.5)
        plt.plot([min(actuals), max(actuals)], [min(actuals), max(actuals)], 'r--')
        plt.title('Actual vs Predicted Values')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        actual_vs_pred_path = "actual_vs_predicted.png"
        plt.savefig(actual_vs_pred_path)
        plt.close()
        
        # 2. Residuals histogram
        plt.figure(figsize=(10, 6))
        plt.hist(residuals, bins=30, alpha=0.7)
        plt.title('Residuals Distribution')
        plt.xlabel('Residual Value')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        residuals_hist_path = "residuals_histogram.png"
        plt.savefig(residuals_hist_path)
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
        residuals_vs_pred_path = "residuals_vs_predicted.png"
        plt.savefig(residuals_vs_pred_path)
        plt.close()
        
        # 4. QQ Plot for residuals normality check
        plt.figure(figsize=(10, 6))
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('Q-Q Plot of Residuals')
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        qq_plot_path = "residuals_qq_plot.png"
        plt.savefig(qq_plot_path)
        plt.close()
        
        # 5. Prediction Error Distribution
        plt.figure(figsize=(10, 6))
        error_pct = (predictions - actuals) / np.where(actuals == 0, 1e-10, actuals) * 100
        plt.hist(error_pct, bins=30, alpha=0.7, color='green')
        plt.title('Prediction Error Distribution (% Error)')
        plt.xlabel('Percentage Error')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Save the plot
        error_dist_path = "error_distribution.png"
        plt.savefig(error_dist_path)
        plt.close()
        
        # 6. Feature Importance (using permutation importance)
        try:
            from sklearn.inspection import permutation_importance
            
            # Define a prediction function for the model
            def predict_fn(X_samples):
                X_tensor = torch.tensor(X_samples, dtype=torch.float32).to(device)
                with torch.no_grad():
                    return model(X_tensor).cpu().numpy().flatten()
            
            # Calculate permutation importance
            perm_importance = permutation_importance(predict_fn, X_scaled, actuals, n_repeats=5, random_state=42)
            feature_importance = {}
            for i, col in enumerate(X.columns):
                feature_importance[col] = perm_importance.importances_mean[i]
            
            # Sort features by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            features = [x[0] for x in sorted_features]
            importances = [x[1] for x in sorted_features]
            
            plt.figure(figsize=(12, 8))
            plt.barh(features, importances)
            plt.title('Feature Importance (Permutation)')
            plt.xlabel('Importance')
            plt.tight_layout()
            
            # Save the plot
            importance_plot_path = "feature_importance.png"
            plt.savefig(importance_plot_path)
            plt.close()
            
            # Log feature importance to MLflow
            for feature, importance in feature_importance.items():
                mlflow.log_metric(f"importance_{feature}", importance)
            
            # Log the plot
            mlflow.log_artifact(importance_plot_path)
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
        
        # 8. Prediction Error by Actual Value Range
        plt.figure(figsize=(10, 6))
        
        # Create bins for actual values
        bins = pd.cut(actuals, 10)
        bin_means = pd.DataFrame({'actual': actuals, 'error': np.abs(actuals - predictions)}).groupby(bins).mean()
        
        plt.bar(range(len(bin_means)), bin_means['error'], width=0.7)
        plt.xticks(range(len(bin_means)), [f"{b.left:.1f}-{b.right:.1f}" for b in bin_means.index], rotation=45)
        plt.title('Mean Absolute Error by Actual Value Range')
        plt.xlabel('Actual Value Range')
        plt.ylabel('Mean Absolute Error')
        plt.tight_layout()
        
        # Save the plot
        error_by_range_path = "error_by_range.png"
        plt.savefig(error_by_range_path)
        plt.close()
        
        # Log all plots to MLflow
        mlflow.log_artifact(actual_vs_pred_path)
        mlflow.log_artifact(residuals_hist_path)
        mlflow.log_artifact(residuals_vs_pred_path)
        mlflow.log_artifact(qq_plot_path)
        mlflow.log_artifact(error_dist_path)
        mlflow.log_artifact(corr_plot_path)
        mlflow.log_artifact(error_by_range_path)
        
    except Exception as e:
        print(f"Warning: Could not create visualizations: {e}")