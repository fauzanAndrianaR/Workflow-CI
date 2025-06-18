FROM python:3.12.7-slim

WORKDIR /app

COPY conda.yaml .
COPY modelling.py .
COPY dataset_preprocessing/ ./dataset_preprocessing/

RUN pip install --no-cache-dir mlflow==2.19.0 scikit-learn pandas numpy matplotlib seaborn

ENV MLFLOW_TRACKING_URI=file:///app/mlruns
ENV PYTHONPATH=/app

RUN mkdir -p /app/mlruns

EXPOSE 5000

CMD ["python", "modelling.py"]