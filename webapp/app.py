"""Web app to try the Ride Classifier API: upload a ride, see the prediction.

The app never loads the model itself: it calls the API, like any other client would.
Run locally:  uv run streamlit run webapp/app.py
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")
# Address of the API as seen from the browser (inside Docker, API_URL is http://api:8000)
API_PUBLIC_URL = os.environ.get("API_PUBLIC_URL", API_URL)
EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
SENSORS = {
    "Accelerometer (m/s²)": ["ax", "ay", "az"],
    "Gyroscope (°/s)": ["rx", "ry", "rz"],
}

st.set_page_config(page_title="Ride Classifier", page_icon="🛴", layout="wide")


def api_status() -> dict | None:
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.json() if response.ok else None
    except requests.RequestException:
        return None


def predict(csv_bytes: bytes, filename: str) -> requests.Response:
    return requests.post(f"{API_URL}/predict", files={"file": (filename, csv_bytes, "text/csv")}, timeout=60)


with st.sidebar:
    st.header("API")
    st.caption(API_PUBLIC_URL)
    health = api_status()
    if health:
        model = health["model"]
        st.success(f"Ready · model **{model['name']}** v{model['version']} (@{model['alias']})")
    else:
        st.error("API unreachable or no model loaded")
    st.markdown(f"[Swagger documentation]({API_PUBLIC_URL}/docs)")

st.title("🛴 Ride Classifier")
st.write(
    "Upload the sensor measurements of **one ride** (CSV with columns "
    "`time_index, ax, ay, az, rx, ry, rz`) to classify it as "
    "**Safe, Safe overall, Bad weather, Tandem or Reckless**."
)

source = st.radio("Ride", ["Upload a CSV", "Use an example"], horizontal=True)
csv_bytes, filename = None, None
if source == "Upload a CSV":
    uploaded = st.file_uploader("CSV file", type="csv")
    if uploaded:
        csv_bytes, filename = uploaded.getvalue(), uploaded.name
else:
    examples = sorted(p.name for p in EXAMPLES_DIR.glob("*.csv"))
    if examples:
        filename = st.selectbox("Example ride", examples)
        csv_bytes = (EXAMPLES_DIR / filename).read_bytes()
    else:
        st.warning(f"No example found in {EXAMPLES_DIR}")

if csv_bytes:
    ride = pd.read_csv(io.BytesIO(csv_bytes))
    st.caption(f"{len(ride)} measurements · about {len(ride) / 10 / 60:.1f} minutes")

    if st.button("Classify this ride", type="primary"):
        with st.spinner("Asking the API..."):
            try:
                response = predict(csv_bytes, filename)
            except requests.RequestException as error:
                st.error(f"Could not reach the API: {error}")
                st.stop()

        if response.ok:
            result = response.json()
            left, right = st.columns([1, 2])
            left.metric("Predicted class", result["label"])
            left.caption(f"Model v{result['model']['version']} · run {result['model']['run_id'][:8]}")
            probabilities = pd.DataFrame(
                {"class": list(result["probabilities"]), "probability": list(result["probabilities"].values())}
            )
            right.bar_chart(probabilities, x="class", y="probability", horizontal=True, sort="-probability")
        else:
            st.error(f"API error {response.status_code}: {response.json().get('detail')}")

    with st.expander("Sensor signals", expanded=False):
        if "time_index" in ride.columns:
            signals = ride.set_index("time_index")
            for title, columns in SENSORS.items():
                present = [c for c in columns if c in signals.columns]
                if present:
                    st.write(title)
                    st.line_chart(signals[present])
