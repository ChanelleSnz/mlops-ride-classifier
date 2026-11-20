"""Feature store pipeline: register the definitions, build the training set, fill the online store.

Run from the project root, after `uv run python -m ride_classifier.features`:
    uv run --project feature_repo python feature_repo/build_training_set.py
Output: data/processed/ride_features_training.parquet, read by ride_classifier.feature_model.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from definitions import FEATURE_NAMES, FEATURE_SERVICE, REPO_DIR, definitions
from feast import FeatureStore

ROOT = REPO_DIR.parent
LABELS_PATH = ROOT / "data" / "raw" / "GBG500_labels.csv"
TRAINING_SET_PATH = ROOT / "data" / "processed" / "ride_features_training.parquet"


def build_training_set(store: FeatureStore, ride_ids: pd.Series) -> pd.DataFrame:
    """Point-in-time join: the latest features of each ride known at 'now'.

    Labels are not stored in Feast: they stay in GBG500_labels.csv and are joined by the training code.
    """
    entity_df = pd.DataFrame({"ride_id": ride_ids.to_numpy(), "event_timestamp": pd.Timestamp.now(tz="UTC")})
    service = store.get_feature_service(FEATURE_SERVICE)
    training = store.get_historical_features(entity_df=entity_df, features=service).to_df()

    incomplete = training[FEATURE_NAMES].isna().any(axis=1)
    if incomplete.any():
        missing = training.loc[incomplete, "ride_id"].head().tolist()
        raise ValueError(f"{int(incomplete.sum())} rides have no features in the store, e.g. {missing}")
    return training.drop(columns="event_timestamp").sort_values("ride_id").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels", type=Path, default=LABELS_PATH, help="GBG500_labels.csv (rides to include)")
    parser.add_argument("--out", type=Path, default=TRAINING_SET_PATH)
    args = parser.parse_args()

    store = FeatureStore(repo_path=str(REPO_DIR))
    store.apply(definitions())
    print(f"Applied feature views: {[view.name for view in store.list_feature_views()]}")

    labels = pd.read_csv(args.labels)
    training = build_training_set(store, labels["RIDE_ID"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    training.to_parquet(args.out, index=False)
    print(f"Training set: {training.shape[0]} rides x {len(FEATURE_NAMES)} features -> {args.out}")

    # Online store: the latest features of every ride, for low-latency lookups by ride_id
    store.materialize_incremental(end_date=datetime.now(UTC))
    first = training["ride_id"].iloc[0]
    online = store.get_online_features(
        features=store.get_feature_service(FEATURE_SERVICE), entity_rows=[{"ride_id": first}]
    ).to_dict()
    print(f"Online lookup for ride {first[:12]}…: acc_mag_mean = {online['acc_mag_mean'][0]:.3f}")


if __name__ == "__main__":
    main()
