import os
import pandas as pd
import joblib
import numpy as np
import mlflow
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from dotenv import load_dotenv


# Load variables from .env file
load_dotenv()

mlflowURI = os.getenv("MLFLOW_TRACKING_URI")
weight_path = os.getenv("MODEL_WIGHTS_PATH")
dataset_path = os.getenv("DATASET_PATH")
target_column = os.getenv("TARGET_COLUMN")
experiment_name = os.getenv("EXPERIMENT_NAME")

# Set MLflow tracking URI
mlflow.set_tracking_uri(mlflowURI)

# Set the experiment name
if mlflow.get_experiment_by_name(experiment_name) is None:
    mlflow.create_experiment(experiment_name)
mlflow.set_experiment(experiment_name)

# === Step 1: Load the trained model from the pickle file ===
model = joblib.load(weight_path)

# === Step 2: Load the dataset from CSV ===
data = pd.read_csv(dataset_path)

# Assuming 'Weight' is the target column
# Modify this based on your actual target column name
target_column = target_column

# Split features and target
X = data.drop(columns=[target_column])
y_true = data[target_column]

# === Step 3: Make predictions ===
predictions = model.predict(X)

print("Predictions:")
print(predictions)

# === Step 4: Calculate evaluation metrics ===
mae = mean_absolute_error(y_true, predictions)
mse = mean_squared_error(y_true, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_true, predictions)

# Log metrics to MLflow
# with mlflow.start_run(run_name="model_evaluation"):
mlflow.set_tracking_uri(mlflowURI)

# Set the experiment
mlflow.set_experiment(experiment_name)

with mlflow.start_run():
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    
    # Log the model to MLflow
    # mlflow.sklearn.log_model(model, "model")

    #fix for the trace issue and trying to regiter in the model registry

    # mlflow.sklearn.log_model(model, artifact_path="model")
    # mlflow.sklearn.log_model(model, artifact_path="model")


    
    # Log model parameters
    model_params = model.get_params()
    for param_name, param_value in model_params.items():
        mlflow.log_param(param_name, param_value)

# === Step 5: Output results ===
print("\nEvaluation Metrics:")
print(f"MAE (Mean Absolute Error):       {mae:.4f}")
print(f"MSE (Mean Squared Error):        {mse:.4f}")
print(f"RMSE (Root Mean Squared Error):  {rmse:.4f}")
print(f"R-Squared (R²):                  {r2:.4f}")
print(f"\nMetrics and model have been logged to MLflow at: {mlflowURI}")

# Combine with original data for reference
output = data.copy()
output['Prediction'] = predictions
output['Absolute Error'] = abs(output[target_column] - predictions)

# Save to CSV
# output.to_csv('predictions_output.csv', index=False)
# print("\nPredictions saved to predictions_output.csv")
