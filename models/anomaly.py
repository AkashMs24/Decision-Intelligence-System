# models/anomaly.py
import pandas as pd
import numpy as np


def detect_anomalies(df: pd.DataFrame, max_results: int = 8) -> list[dict]:
    """
    Detect revenue anomalies using a Z-score approach (>2.5σ triggers an alert).
    Falls back gracefully when the series is too short or has zero variance.

    Returns a list of dicts: Event | Date | Impact | Severity
    """
    df = df.copy().sort_values("date").reset_index(drop=True)

    series = pd.to_numeric(df["revenue"], errors="coerce").fillna(0)
    mean = series.mean()
    std  = series.std()

    # Not enough variance to compute meaningful z-scores
    if std == 0 or len(series) < 5:
        return []

    df["z_score"] = (series - mean) / std
    anomalies = df[df["z_score"].abs() > 2.5].copy()

    result = []
    for _, row in anomalies.iterrows():
        z = abs(float(row["z_score"]))
        severity = "High" if z > 3.5 else "Medium"
        result.append(
            {
                "Event":    f"Revenue anomaly (z={z:.1f}σ)",
                "Date":     str(row["date"]),
                "Impact":   f"₹{float(row['revenue']) / 100_000:.2f}L",
                "Severity": severity,
            }
        )

    # Sort highest |z| first so the most extreme anomalies appear at the top
    result.sort(key=lambda x: float(x["Event"].split("z=")[1].rstrip("σ)")), reverse=True)
    return result[:max_results]
