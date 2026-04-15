# models/anomaly.py
import pandas as pd

def detect_anomalies(df):
    df = df.copy()
    if len(df) < 5 or 'revenue' not in df.columns:
        return []
    
    mean = df['revenue'].mean()
    std = df['revenue'].std()
    if std == 0:
        return []
    
    df['z_score'] = (df['revenue'] - mean) / std
    anomalies = df[abs(df['z_score']) > 2.5].copy()
    
    result = []
    for _, row in anomalies.iterrows():
        z = abs(row['z_score'])
        sev = "High" if z > 3 else "Medium"
        result.append({
            "Event": f"Revenue anomaly (z={z:.1f}σ)",
            "Date": str(row["date"]),
            "Impact": f"₹{row['revenue']/100_000:.2f}L",
            "Severity": sev
        })
    return result[:8]
