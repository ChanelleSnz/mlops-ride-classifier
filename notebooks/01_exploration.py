"""Exploration des données GBG500."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
SENSORS = ["ax", "ay", "az", "rx", "ry", "rz"]

rides = pd.read_parquet(RAW / "GBG500.parquet")
labels = pd.read_csv(RAW / "GBG500_labels.csv")
print("Taille :", rides.shape, labels.shape)

print("\n--- Répartition des classes ---")
print(labels["LABEL"].value_counts())

print("\n--- Valeurs manquantes et doublons ---")
missing = rides[SENSORS].isna().any(axis=1)
print("Lignes sans mesure :", missing.sum(), f"({missing.mean():.1%})")
print("Trajets concernés :", rides.loc[missing, "ride_id"].nunique())
print("Doublons :", rides.duplicated(["ride_id", "time_index"]).sum())

print("\n--- Durée des trajets par classe (secondes) ---")
duration = rides.groupby("ride_id").size().div(10).rename("seconds")
duration = labels.set_index("RIDE_ID").join(duration)
print(duration.groupby("LABEL")["seconds"].describe().round(0))

fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)
for ax, (label, group) in zip(axes, labels.groupby("LABEL"), strict=True):
    ride_id = group["RIDE_ID"].iloc[0]
    ride = rides[rides["ride_id"] == ride_id].set_index("time_index")
    ride[SENSORS].plot(ax=ax, linewidth=0.6, legend=False)
    ax.set_title(label)
axes[0].legend(SENSORS, ncol=6, loc="upper right")
plt.tight_layout()
out = ROOT / "docs" / "eda_one_ride_per_class.png"
plt.savefig(out, dpi=120)
print("\nGraphique enregistré :", out)
