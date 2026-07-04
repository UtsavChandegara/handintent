# Dockerfile
# HandIntent Training Container
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy training script
COPY train_bigru_cloud.py .

# Run training when container starts
CMD ["python3", "train_bigru_cloud.py"]
