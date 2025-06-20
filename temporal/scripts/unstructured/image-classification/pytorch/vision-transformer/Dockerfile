FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY src/ /app/src/
COPY weights/ /app/weights/
COPY datasets/ /app/datasets/

# Set the pip cache directory to a volume
ENV PIP_CACHE_DIR=/root/.cache/pip

# Run the evaluation script
CMD ["python3", "src/eval.py"]