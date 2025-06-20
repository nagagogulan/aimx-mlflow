import os
import pandas as pd
import joblib
import mlflow
from dotenv import load_dotenv
from sklearn.metrics import roc_auc_score, roc_curve, log_loss, cohen_kappa_score
from sklearn.metrics import brier_score_loss

# Load variables from .env file
load_dotenv()

mlflowURI = os.getenv("MLFLOW_TRACKING_URI")
weight_path = os.getenv("MODEL_WIGHTS_PATH")
dataset_path = os.getenv("DATASET_PATH")
target_column = os.getenv("TARGET_COLUMN")
experiment_name = os.getenv("EXPERIMENT_NAME")

# Step 1: Load the dataset
df = pd.read_csv(dataset_path)

# Step 2: Separate features and target
X = df.drop(columns=[target_column])
y = df[target_column]

# Step 3: Preprocess if needed
# Convert object-type features to category
for col in X.select_dtypes(include='object').columns:
    X[col] = X[col].astype('category')

# Step 4: Load your trained model
model = joblib.load(weight_path)  # Update if filename differs

# Step 5: Predict
predictions = model.predict(X)

# Get prediction probabilities for ROC-AUC calculation
# Check if the model has predict_proba method
try:
    # For multi-class classification
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X)
    # For LightGBM models that use predict directly for probabilities
    else:
        y_prob = predictions
        # If predictions are not probabilities (i.e., already binary)
        if len(predictions.shape) == 1 and (predictions.max() <= 1.0001 and predictions.min() >= -0.0001):
            # Reshape to match sklearn's predict_proba output format
            y_prob = pd.DataFrame({
                0: 1 - predictions,
                1: predictions
            }).values
except:
    y_prob = None

# Step 6: Compare with actuals
results = pd.DataFrame({
    "Actual": y,
    "Predicted": predictions
})

print(results)

# Step 7: Calculate Evaluation Metrics
# Count correct predictions (TP + TN)
correct_predictions = (results["Actual"] == results["Predicted"]).sum()
# Total number of predictions (TP + TN + FP + FN)
total_predictions = len(results)
# Calculate accuracy using the formula (TP + TN) / (TP + TN + FP + FN)
accuracy = correct_predictions / total_predictions

# Define the positive class
positive_class = "TruePositive"

# True Positives: predicted positive and actually positive
true_positives = ((results["Predicted"] == positive_class) & 
                 (results["Actual"] == positive_class)).sum()

# False Positives: predicted positive but actually negative
false_positives = ((results["Predicted"] == positive_class) & 
                  (results["Actual"] != positive_class)).sum()

# False Negatives: predicted negative but actually positive
false_negatives = ((results["Predicted"] != positive_class) & 
                  (results["Actual"] == positive_class)).sum()

# Calculate precision using the formula: TP / (TP + FP)
precision = 0  # Default value
if (true_positives + false_positives) > 0:  # Avoid division by zero
    precision = true_positives / (true_positives + false_positives)

# Calculate recall using the formula: TP / (TP + FN)
recall = 0  # Default value
if (true_positives + false_negatives) > 0:  # Avoid division by zero
    recall = true_positives / (true_positives + false_negatives)

# Calculate specificity (TNR) using the formula: TN / (TN + FP)
# True Negatives: predicted negative and actually negative
true_negatives = ((results["Predicted"] != positive_class) & 
                 (results["Actual"] != positive_class)).sum()
                 
specificity = 0  # Default value
if (true_negatives + false_positives) > 0:  # Avoid division by zero
    specificity = true_negatives / (true_negatives + false_positives)

# Calculate F1-score using the formula: 2 * (Precision * Recall) / (Precision + Recall)
f1_score = 0  # Default value
if (precision + recall) > 0:  # Avoid division by zero
    f1_score = 2 * (precision * recall) / (precision + recall)

# Calculate Youden's Index (J-statistic) using the formula: Sensitivity + Specificity - 1
# Note: Sensitivity is the same as Recall
youdens_index = recall + specificity - 1

# Calculate Matthews Correlation Coefficient (MCC)
mcc_numerator = (true_positives * true_negatives) - (false_positives * false_negatives)
mcc_denominator = ((true_positives + false_positives) * 
                   (true_positives + false_negatives) * 
                   (true_negatives + false_positives) * 
                   (true_negatives + false_negatives))

# Default value
mcc = 0
# Avoid division by zero and square root of negative number
if mcc_denominator > 0:
    mcc = mcc_numerator / (mcc_denominator ** 0.5)

# Calculate AUC-ROC (Area Under the Receiver Operating Characteristic Curve)
auc_roc = 0  # Default value
try:
    # For binary classification or if classes are encoded as 0 and 1            mlflow.log_metric(f"model_{metric_name}", value)
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

is these belongs to tabular regression,tensorflow,custom
    if y_prob is not None:
        if len(y_prob.shape) > 1 and y_prob.shape[1] == 2:
            # Binary classification: use probabilities of the positive class
            auc_roc = roc_auc_score(y, y_prob[:, 1])
        else:
            # Multi-class: use one-vs-rest approach (ovr)
            auc_roc = roc_auc_score(pd.get_dummies(y), y_prob, multi_class='ovr')
except Exception as e:
    print(f"Warning: Could not calculate AUC-ROC: {e}")

# Calculate Logarithmic Loss (LogLoss)
# Formula: -(1/N) Σ [y_i log(ŷ_i) + (1-y_i) log(1-ŷ_i)]
logloss = 0  # Default value
try:
    if y_prob is not None:
        # Calculate log loss using scikit-learn's log_loss function
        logloss = log_loss(y, y_prob)
except Exception as e:
    print(f"Warning: Could not calculate LogLoss: {e}")

# Calculate Brier Score: (1/N) Σ (y_i - ŷ_i)^2
brier_score = 0  # Default value
try:
    if y_prob is not None and y_prob.shape[1] == 2:
        # Convert target to binary format for binary classification
        y_binary = (y == 1).astype(int)
        # Calculate Brier score using scikit-learn's brier_score_loss function
        brier_score = brier_score_loss(y_binary, y_prob[:, 1])
except Exception as e:
    print(f"Warning: Could not calculate Brier Score: {e}")

# Calculate Cohen's Kappa (inter-annotator agreement)
cohens_kappa = 0  # Default value
try:
    # Calculate Cohen's kappa using scikit-learn's cohen_kappa_score function
    cohens_kappa = cohen_kappa_score(y, predictions)
except Exception as e:
    print(f"Warning: Could not calculate Cohen's Kappa: {e}")

# Calculate Balanced Accuracy: (Sensitivity + Specificity) / 2
# Note: Sensitivity is the same as Recall
balanced_accuracy = (recall + specificity) / 2

# LightGBM-specific metrics
# Feature importance
feature_importance = {}
if hasattr(model, 'feature_importances_'):
    for i, importance in enumerate(model.feature_importances_):
        feature_importance[X.columns[i]] = float(importance)

print(f"\nAccuracy: {accuracy:.4f} ({correct_predictions}/{total_predictions})")
print(f"Precision: {precision:.4f} ({true_positives}/{true_positives + false_positives})")
print(f"Recall: {recall:.4f} ({true_positives}/{true_positives + false_negatives})")
print(f"Specificity: {specificity:.4f} ({true_negatives}/{true_negatives + false_positives})")
print(f"F1-score: {f1_score:.4f}")
print(f"Youden's Index: {youdens_index:.4f}")
print(f"Matthews Correlation Coefficient: {mcc:.4f}")
print(f"AUC-ROC: {auc_roc:.4f}")
print(f"LogLoss: {logloss:.4f}")
print(f"Brier Score: {brier_score:.4f}")
print(f"Cohen's Kappa: {cohens_kappa:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy:.4f}")

# Step 8: Log metrics to MLflow
# Set the MLflow tracking URI
mlflow.set_tracking_uri(mlflowURI)

# Set the experiment
mlflow.set_experiment(experiment_name)

# Start an MLflow run
with mlflow.start_run():
    # Log the metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("specificity", specificity)
    mlflow.log_metric("f1_score", f1_score)
    mlflow.log_metric("youdens_index", youdens_index)
    mlflow.log_metric("mcc", mcc)
    mlflow.log_metric("auc_roc", auc_roc)
    mlflow.log_metric("logloss", logloss)
    mlflow.log_metric("brier_score", brier_score)
    mlflow.log_metric("cohens_kappa", cohens_kappa)
    mlflow.log_metric("balanced_accuracy", balanced_accuracy)
    
    # Log feature importance
    for feature, importance in feature_importance.items():
        mlflow.log_metric(f"importance_{feature}", importance)
    
    # You can log additional information if needed
    mlflow.log_param("model_path", weight_path)
    mlflow.log_param("dataset_path", dataset_path)
    
    print(f"Metrics successfully logged to MLflow experiment: {experiment_name}")