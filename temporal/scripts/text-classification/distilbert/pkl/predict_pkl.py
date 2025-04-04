import pickle
import torch
from transformers import AutoTokenizer

# Load the model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Ensure the model is in evaluation mode
model.eval()

# Prediction function
def predict(text):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # Forward pass through the model
    with torch.no_grad():
        outputs = model(**inputs)

    # Convert logits to predicted class
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    
    return predicted_class

# Example usage
text = "This is not so great product!"
prediction = predict(text)
print(f"Predicted class: {prediction}")  # 1 (Positive) or 0 (Negative)
