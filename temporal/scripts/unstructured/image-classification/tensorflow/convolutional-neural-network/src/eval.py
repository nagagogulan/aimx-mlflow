import os
from tensorflow.keras.models import load_model
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

weight_path = os.getenv("MODEL_WIGHTS_PATH")

model = load_model(weight_path)
model.summary()