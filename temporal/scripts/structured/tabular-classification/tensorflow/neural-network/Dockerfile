FROM tensorflow/tensorflow:2.9.0

# Set the working directory
WORKDIR /app

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