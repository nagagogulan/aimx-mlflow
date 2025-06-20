FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install build dependencies for LightGBM
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libboost-dev \
    libboost-system-dev \
    libboost-filesystem-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

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