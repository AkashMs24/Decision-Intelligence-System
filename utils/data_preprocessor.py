# utils/data_preprocessor.py
import pandas as pd
import numpy as np

def transform_user_data(raw: pd.DataFrame, filename="Unknown") -> pd.DataFrame:
    df = raw.copy()
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Auto-detect columns
    date_col = next((c for c in df.columns if any(k in c for k in ["date", "time", "invoice", "day"])), None)
    rev_col = next((c for c in df.columns if any(k in c for k in ["revenue", "sales", "amount", "price", "income", "spending"])), None)
    cust_col = next((c for c in df.columns if any(k in c for k in ["customer", "user", "client", "id"])), None)

    # Special handling for Mall_Customers type dataset
    if "income" in df.columns and "spending_score" in df.columns:
        df["revenue"] = df["income"] * (df["spending_score"] / 10)   # synthetic revenue
        df["customers"] = np.random.randint(50, 300, len(df))
    elif rev_col:
        df["revenue"] = pd.to_numeric(df[rev_col], errors="coerce").fillna(0)
        df["customers"] = pd.to_numeric(df.get(cust_col, len(df)), errors="coerce").fillna(100)
    else:
        df["revenue"] = np.random.uniform(10000, 100000, len(df))
        df["customers"] = np.random.randint(50, 300, len(df))

    if date_col:
        df["date"] = pd.to_datetime(df[date_col], errors='coerce').dt.date
    else:
        df["date"] = pd.date_range(start="2024-01-01", periods=len(df)).date

    df["marketing_spend"] = df["revenue"] * 0.20
    df["churn"] = (df["customers"].pct_change() < 0).astype(int).fillna(0)

    df = df[["date", "revenue", "customers", "marketing_spend", "churn"]].reset_index(drop=True)

    # Attach metadata for LLM
    df.attrs['filename'] = filename
    df.attrs['num_rows'] = len(df)
    df.attrs['columns'] = list(raw.columns)

    return df
