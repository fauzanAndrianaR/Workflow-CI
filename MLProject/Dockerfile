# MLProject/Dockerfile
FROM python:3.12.7-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY conda.yaml .
COPY modelling.py .
COPY dataset_preprocessing/ ./dataset_preprocessing/

# Install Python dependencies
RUN pip install --no-cache-dir mlflow==2.19.0 scikit-learn pandas numpy matplotlib seaborn

# Set environment variables for MLflow
ENV MLFLOW_TRACKING_URI=file:///app/mlruns
ENV PYTHONPATH=/app

# Create necessary directories
RUN mkdir -p /app/mlruns

# Expose MLflow port
EXPOSE 5000

# Default command
CMD ["python", "modelling.py"]