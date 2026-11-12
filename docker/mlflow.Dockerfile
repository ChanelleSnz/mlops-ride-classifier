# MLflow tracking server with PostgreSQL (metadata) and MinIO/S3 (artifacts) support
FROM python:3.12-slim

RUN pip install --no-cache-dir \
    mlflow==3.18.0 \
    psycopg2-binary==2.9.13 \
    boto3==1.44.1

COPY docker/create_bucket.py /app/create_bucket.py

EXPOSE 5000
