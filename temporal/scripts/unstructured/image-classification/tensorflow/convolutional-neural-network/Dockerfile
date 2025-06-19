FROM python:3.9

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /app/src/
COPY weights/ /app/weights/
COPY datasets/ /app/datasets/

ENV PIP_CACHE_DIR=/root/.cache/pip

CMD ["python3", "-u", "src/eval.py"]
