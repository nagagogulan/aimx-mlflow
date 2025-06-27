# FROM python:3.9

# # Set the working directory
# WORKDIR /app

# # Copy the requirements file
# COPY requirements.txt .

# # Install dependencies
# # RUN pip install --no-cache-dir -r requirements.txt
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy the source code
# COPY src/ /app/src/

# # Copy the weights
# COPY weights/ /app/weights/

# # Copy datasets
# COPY datasets/ /app/datasets/

# # Set the pip cache directory to a volume
# ENV PIP_CACHE_DIR=/root/.cache/pip

# # Set environment variables
# # ENV MODEL_NAME=distilbert/distilbert-base-uncased-finetuned-sst-2-english
# # ENV DEVICE=0
# # ENV MODEL_WIGHTS_PATH=./weights

# # Command to run the application
# CMD ["python3", "src/eval.py"]



FROM python:3.9
 
# Set the working directory
WORKDIR /app

# Install jq for JSON editing
RUN apt-get update && apt-get install -y jq
 
# Copy the requirements file
COPY requirements.txt .
 
# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
 
# Copy the source code
COPY src/ /app/src/
 
# Copy the weights
COPY weights/ /app/weights/
 
# Copy datasets
COPY datasets/ /app/datasets/
 
# Set the pip cache directory to a volume
ENV PIP_CACHE_DIR=/root/.cache/pip
 
# # Command to run the application
# CMD ["python3", "src/eval.py"]


# CMD ["python3", "-u", "src/eval.py"]
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh
CMD ["./entrypoint.sh"]


