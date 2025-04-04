from transformers import AutoModelForSequenceClassification
import pickle

# Load a pretrained model
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Save the model to a .pkl file
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

