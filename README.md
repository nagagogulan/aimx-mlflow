# AIMx-ML

1. Unstructured Data (Image-Based Inference - Classification Model)

Model Type: Image Classification/Segmentation
Model Weight: [Specify weight file, e.g., model.pth, model.onnx, model.h5]
Model Framework: [TensorFlow / PyTorch / ONNX]
Model Architecture: [ResNet-50 / EfficientNet-B4 / Vision Transformer (ViT) / Custom]
Required CPU: Minimum 8 vCPUs (for non-GPU fallback)
Required GPU (optional): A100 (40GB), partitioned access of 10GB-20GB (depending on model complexity)
Required RAM: 16GB (for optimal batch processing)

2. Structured Data (Tabular-Based Model - Classification/Regression)

Model Type: Tabular Classification / Regression
Model Weight: [Specify weight file, e.g., model.pkl, model.joblib]
Model Framework: [XGBoost / LightGBM / Scikit-learn / TensorFlow / PyTorch]
Model Architecture: [Tree-based (XGBoost, LightGBM) / Neural Network (MLP, TabNet) / Custom]
Required CPU: Minimum 4 vCPUs (for batch processing)
Required GPU: Not mandatory but optional for deep learning-based models (e.g., MLPs)
Required RAM: 8GB (scalable based on dataset size)

🔹 Keras .h5 Models:

Brain Tumor Detection - Neuroscan
https://github.com/deepraj21/Neuroscan

Face Mask Detection
https://github.com/Furkan-Gulsen/Face-Mask-Detection

🔹 Scikit-learn .pkl Models:

PredictiX - Multi-Disease Classifier
https://github.com/hallowshaw/PredictiX

🔹 ONNX Model Zoo (Image Classification):

ONNX Pre-trained Models
https://github.com/onnx/models

🔹 Conversion Tools:

ONNXMLTools GitHub
https://github.com/onnx/onnxmltools

TensorFlow to ONNX
https://github.com/onnx/tensorflow-onnx

Sklearn to ONNX
https://github.com/onnx/sklearn-onnx
