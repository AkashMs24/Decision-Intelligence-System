# utils/data_loader.py
import pandas as pd
import os

def load_data(path: str = "data/sample_data.csv") -> pd.DataFrame:
    """
    Load CSV data from a given path.
    Falls back to a minimal synthetic dataset if file is missing.
    """
    if not os.path.exists(path):
        # Generate synthetic fallback data so the app never crashes on startup
        import numpy as np
        dates = pd.date_range(start="2024-01-01", periods=90, freq="D")
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "date": dates.date,
            "sales_amount": rng.uniform(45_000, 220_000, 90),
            "total_orders": rng.integers(130, 870, 90),
            "customer_count": rng.integers(180, 1050, 90),
            "average_order_value": rng.uniform(140, 800, 90),
        })
        return df

    df = pd.read_csv(path, encoding="ISO-8859-1")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.dropna(subset=["date"])
    return df
