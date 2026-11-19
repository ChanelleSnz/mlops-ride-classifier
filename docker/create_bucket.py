"""Create the MLflow artifact bucket in MinIO if it does not exist yet.

Runs before the MLflow server starts. Replaces the `minio/mc` image,
which was removed from Docker Hub in September 2026.
"""

import os
import time

import boto3

BUCKET = os.environ.get("MLFLOW_BUCKET", "mlflow")
s3 = boto3.client("s3", endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"])

for attempt in range(1, 31):
    try:
        existing = {b["Name"] for b in s3.list_buckets().get("Buckets", [])}
        if BUCKET in existing:
            print(f"Bucket '{BUCKET}' already exists")
        else:
            s3.create_bucket(Bucket=BUCKET)
            print(f"Bucket '{BUCKET}' created")
        break
    except Exception as error:  # MinIO may still be starting
        print(f"Waiting for MinIO ({attempt}/30): {error}")
        time.sleep(2)
else:
    raise SystemExit("MinIO is unreachable, MLflow cannot start")
