# FROM python:3.9-slim

# # Set the working directory
# WORKDIR /app

# # Copy the requirements file
# COPY requirements.txt .

# # Install dependencies
# # RUN pip install --no-cache-dir -r requirements.txt
# RUN pip install -r requirements.txt

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








FROM python:3.9-slim

# Install system dependencies for Docker and Python
RUN apt-get update && apt-get install -y \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg \
    lsb-release \
    iptables \
    uidmap \
    gnupg2 \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Add Docker's GPG key and repo
RUN mkdir -m 0755 -p /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
    https://download.docker.com/linux/debian $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
    > /etc/apt/sources.list.d/docker.list

# Install Docker CLI and Daemon
RUN apt-get update && apt-get install -y \
    docker-ce \
    docker-ce-cli \
    containerd.io \
    docker-buildx-plugin \
    docker-compose-plugin

# Set working directory
WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ /app/src/
COPY weights/ /app/weights/
COPY datasets/ /app/datasets/

# Set environment variable
ENV PIP_CACHE_DIR=/root/.cache/pip

# Start Docker daemon in background and run Python app
CMD dockerd & sleep 5 && python3 src/eval.py
