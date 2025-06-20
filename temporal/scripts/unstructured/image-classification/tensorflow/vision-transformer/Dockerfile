FROM tensorflow/tensorflow:2.9.0-gpu

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

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

# Command to run the application
CMD ["python3", "src/eval.py"]
