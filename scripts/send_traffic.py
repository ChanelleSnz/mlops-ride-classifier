"""Send prediction requests to the API to fill the monitoring dashboards.

Usage: uv run python scripts/send_traffic.py [--url http://localhost:8000] [--n 50]
Sends random real rides from data/raw (and a few invalid files to show errors).
"""

from __future__ import annotations

import argparse
import io
import random
import time

import httpx
import pandas as pd

from ride_classifier.data import RAW_DIR


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--error-rate", type=float, default=0.1)
    args = parser.parse_args()

    rides = pd.read_parquet(RAW_DIR / "GBG500.parquet")
    ride_ids = rides["ride_id"].unique().tolist()
    with httpx.Client(base_url=args.url, timeout=60) as client:
        for i in range(1, args.n + 1):
            if random.random() < args.error_rate:
                content = b"time_index,ax\n0,1.0\n"  # missing columns -> 422
            else:
                ride = rides[rides["ride_id"] == random.choice(ride_ids)].drop(columns="ride_id")
                buffer = io.StringIO()
                ride.to_csv(buffer, index=False)
                content = buffer.getvalue().encode()
            response = client.post("/predict", files={"file": ("ride.csv", content, "text/csv")})
            label = response.json().get("label", "-") if response.is_success else ""
            print(f"{i:>3}/{args.n}  {response.status_code}  {label}")
            time.sleep(0.2)


if __name__ == "__main__":
    main()
