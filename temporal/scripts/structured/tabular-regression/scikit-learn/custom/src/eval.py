import os
import pandas as pd
import numpy as np
import pickle
import mlflow
import inspect
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_percentage_error, median_absolute_error
from sklearn.base import BaseEstimator, RegressorMixin

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

# Step 4: Load your trained scikit-learn custom model
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

# Custom model specific metrics
custom_metrics = {}

# Get model type
model_type = type(model).__name__
custom_metrics['model_type'] = model_type

# Check if model is a pipeline
is_pipeline = False
try:
    from sklearn.pipeline import Pipeline
    is_pipeline = isinstance(model, Pipeline)
    if is_pipeline:
        custom_metrics['is_pipeline'] = True
        custom_metrics['pipeline_steps'] = len(model.steps)
        # Get the final estimator
        final_estimator = model.steps[-1][1]
        custom_metrics['final_estimator_type'] = type(final_estimator).__name__
except Exception as e:
    print(f"Warning: Could not check if model is a pipeline: {e}")

# Get model parameters
if hasattr(model, 'get_params'):
    params = model.get_params()
    for param_name, param_value in params.items():
        if isinstance(param_value, (int, float, str, bool)):
            custom_metrics[f"param_{param_name}"] = param_value

# Check for feature importance
if hasattr(model, 'feature_importances_'):
    feature_importance = {feature: importance for feature, importance in 
                         zip(X.columns, model.feature_importances_)}
    
    # Sort feature importance
    feature_importance = dict(sorted(feature_importance.items(), 
                                    key=lambda item: item[1], 
                                    reverse=True))
elif is_pipeline and hasattr(final_estimator, 'feature_importances_'):
    feature_importance = {feature: importance for feature, importance in 
                         zip(X.columns, final_estimator.feature_importances_)}
    
    # Sort feature importance
    feature_importance = dict(sorted(feature_importance.items(), 
                                    key=lambda item: item[1], 
                                    reverse=True))
else:
    feature_importance = {}

# Check for coefficients (linear models)
if hasattr(model, 'coef_'):
    if len(model.coef_.shape) == 1:
        coef_dict = {feature: coef for feature, coef in zip(X.columns, model.coef_)}
        custom_metrics['coef_mean'] = float(np.mean(np.abs(model.coef_)))
        custom_metrics['coef_std'] = float(np.std(model.coef_))
        custom_metrics['coef_min'] = float(np.min(model.coef_))
        custom_metrics['coef_max'] = float(np.max(model.coef_))
        
        # Sort coefficients by absolute value
        coef_dict = {k: v for k, v in sorted(coef_dict.items(), 
                                            key=lambda item: abs(item[1]), 
                                            reverse=True)}
        
        # Store top coefficients
        for i, (feature, coef) in enumerate(coef_dict.items()):
            if i < 10:  # Store top 10
                custom_metrics[f"top_coef_{i+1}_feature"] = feature
                custom_metrics[f"top_coef_{i+1}_value"] = float(coef)
elif is_pipeline and hasattr(final_estimator, 'coef_'):
    if len(final_estimator.coef_.shape) == 1:
        coef_dict = {feature: coef for feature, coef in zip(X.columns, final_estimator.coef_)}
        custom_metrics['coef_mean'] = float(np.mean(np.abs(final_estimator.coef_)))
        custom_metrics['coef_std'] = float(np.std(final_estimator.coef_))
        custom_metrics['coef_min'] = float(np.min(final_estimator.coef_))
        custom_metrics['coef_max'] = float(np.max(final_estimator.coef_))
        
        # Sort coefficients by absolute value
        coef_dict = {k: v for k, v in sorted(coef_dict.items(), 
                                            key=lambda item: abs(item[1]), 
                                            reverse=True)}
        
        # Store top coefficients
        for i, (feature, coef) in enumerate(coef_dict.items()):
            if i < 10:  # Store top 10
                custom_metrics[f"top_coef_{i+1}_feature"] = feature
                custom_metrics[f"top_coef_{i+1}_value"] = float(coef)

# Check for intercept (linear models)
if hasattr(model, 'intercept_'):
    if isinstance(model.intercept_, (int, float)):
        custom_metrics['intercept'] = float(model.intercept_)
    else:
        custom_metrics['intercept'] = float(model.intercept_[0])
elif is_pipeline and hasattr(final_estimator, 'intercept_'):
    if isinstance(final_estimator.intercept_, (int, float)):
        custom_metrics['intercept'] = float(final_estimator.intercept_)
    else:
        custom_metrics['intercept'] = float(final_estimator.intercept_[0])

# Check if model is a custom model (inherits from BaseEstimator)
is_custom_model = isinstance(model, BaseEstimator) and not any(model_type.startswith(prefix) for prefix in 
                                                             ['RandomForest', 'GradientBoosting', 'DecisionTree', 
                                                              'Linear', 'Ridge', 'Lasso', 'ElasticNet', 'SVR', 
                                                              'KNeighbors', 'MLPRegressor'])

if is_custom_model:
    custom_metrics['is_custom_implementation'] = True
    
    # Get all public attributes of the model
    attrs = [attr for attr in dir(model) if not attr.startswith('_') and not callable(getattr(model, attr))]
    
    # Log interesting attributes
    for attr in attrs:
        try:
            value = getattr(model, attr)
            if isinstance(value, (int, float, str, bool)):
                custom_metrics[f"custom_attr_{attr}"] = value
            elif isinstance(value, np.ndarray) and value.size < 10:
                custom_metrics[f"custom_attr_{attr}_mean"] = float(np.mean(value))
                custom_metrics[f"custom_attr_{attr}_std"] = float(np.std(value))
        except:
            pass
    
    # Get methods
    methods = [method for method in dir(model) if callable(getattr(model, method)) and not method.startswith('_')]
    custom_metrics['custom_methods'] = ', '.join(methods)

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

# Print custom model specific metrics
print(f"\nCustom Model Information ({model_type}):")
for metric_name, value in custom_metrics.items():
    if isinstance(value, (int, float, str, bool)):
        print(f"{metric_name}: {value}")

# Print feature importance or coefficients
if feature_importance:
    print("\nTop Feature Importances:")
    for i, (feature, importance) in enumerate(feature_importance.items()):
        if i < 10:  # Show top 10 features
            print(f"{feature}: {importance:.4f}")
elif 'top_coef_1_feature' in custom_metrics:
    print("\nTop Coefficients:")
    for i in range(1, 11):
        if f"top_coef_{i}_feature" in custom_metrics:
            feature = custom_metrics[f"top_coef_{i}_feature"]
            value = custom_metrics[f"top_coef_{i}_value"]
            print(f"{feature}: {value:.4f}")

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
    
    # Log feature importance or coefficients
    if feature_importance:
        for feature, importance in feature_importance.items():
            mlflow.log_metric(f"importance_{feature}", importance)
    
    # Log custom model specific metrics
    for metric_name, value in custom_metrics.items():
        if isinstance(value, (int, float)) and not isinstance(value, dict):
            mlflow.log_metric(f"custom_{metric_name}", value)
        elif isinstance(value, (str, bool)) and not isinstance(value, dict):
            mlflow.log_param(f"custom_{metric_name}", value)
    
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
    
    # 4. Feature importance or coefficients plot
    if feature_importance:
        # Sort feature importances
        sorted_importances = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        features = [x[0] for x in sorted_importances[:10]]  # Top 10 features
        importances = [x[1] for x in sorted_importances[:10]]
        
        plt.figure(figsize=(12, 8))
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
    elif 'top_coef_1_feature' in custom_metrics:
        # Extract top coefficients
        features = []
        coefficients = []
        for i in range(1, 11):
            if f"top_coef_{i}_feature" in custom_metrics:
                features.append(custom_metrics[f"top_coef_{i}_feature"])
                coefficients.append(custom_metrics[f"top_coef_{i}_value"])
        
        if features and coefficients:
            plt.figure(figsize=(12, 8))
            colors = ['blue' if c >= 0 else 'red' for c in coefficients]
            plt.barh(range(len(features)), coefficients, align='center', color=colors)
            plt.yticks(range(len(features)), features)
            plt.title('Top 10 Coefficients by Magnitude')
            plt.xlabel('Coefficient Value')
            plt.ylabel('Feature')
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            plt.tight_layout()
            
            # Save the plot
            coef_plot_path = "coefficient_importance.png"
            plt.savefig(coef_plot_path)
            plt.close()
            
            # Log to MLflow
            with mlflow.start_run(run_id=mlflow.active_run().info.run_id):
                mlflow.log_artifact(coef_plot_path)
    
    # 5. QQ Plot for residuals normality check
    from scipy import stats
    
    plt.figure(figsize=(10, 6))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot of Residuals', fontsize=15)
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    qq_plot_path = "residuals_qq_plot.png"
    plt.savefig(qq_plot_path)
    plt.close()
    
    # 6. Prediction Error Distribution
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
    
    # 7. Feature correlation with residuals
    residual_correlations = {}
    for col in X.columns:
        correlation = np.corrcoef(X[col], residuals)[0, 1]
        residual_correlations[col] = correlation
    
    # Sort by absolute correlation
    sorted_correlations = sorted(residual_correlations.items(), 
                                key=lambda x: abs(x[1]), 
                                reverse=True)
    
    # Plot top correlations
    top_n = min(10, len(sorted_correlations))
    features = [x[0] for x in sorted_correlations[:top_n]]
    correlations = [x[1] for x in sorted_correlations[:top_n]]
    
    plt.figure(figsize=(12, 8))
    colors = ['blue' if c >= 0 else 'red' for c in correlations]
    plt.barh(range(len(features)), correlations, align='center', color=colors)
    plt.yticks(range(len(features)), features)
    plt.title('Feature Correlation with Residuals')
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Feature')
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    residual_corr_path = "residual_correlations.png"
    plt.savefig(residual_corr_path)
    plt.close()
    
    # Log all plots to MLflow
    with mlflow.start_run(run_id=mlflow.active_run().info.run_id):
        mlflow.log_artifact(residuals_plot_path)
        mlflow.log_artifact(scatter_plot_path)
        mlflow.log_artifact(residuals_vs_predicted_path)
        mlflow.log_artifact(qq_plot_path)
        mlflow.log_artifact(error_dist_path)
        mlflow.log_artifact(residual_corr_path)
        if feature_importance:
            mlflow.log_artifact(importance_plot_path)
    
    print("All visualizations created and logged to MLflow")
except Exception as e:
    print(f"Warning: Could not create visualizations: {e}")